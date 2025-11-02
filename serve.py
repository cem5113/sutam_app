#!/usr/bin/env python3
# serve.py — SUTAM tahmin çekirdeği (binary + multiclass, otomatik model seçimi)
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# =========================
# 0) Ayarlar
# =========================
MODELS_DIR = os.environ.get("SUTAM_MODELS_DIR", "models")

BIN_NAME = "sutam_binary_stack_cal.joblib"
BIN_META = "binary_meta.pkl"

# Kategori için sıralı arama (hangisi varsa onu yükle)
CAT_CANDIDATES = [
    "sutam_category_xgb.joblib",   # önerilen
    "sutam_category_lgbm.joblib",  # alternatif
]
CAT_META = "category_meta.pkl"


# =========================
# 1) Model yükleme (güvenli)
# =========================
def _first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(" / ".join(paths))

@lru_cache(maxsize=1)
def _load_models() -> Tuple[object, object, List[str], List[str], List[str]]:
    """
    Modelleri ve meta bilgileri yükler; bulunamazsa açıklayıcı hata verir.
    Dönüş:
      BIN, CAT, BIN_COLS, CAT_COLS, CAT_CLASSES
    """
    try:
        bin_path = os.path.join(MODELS_DIR, BIN_NAME)
        cat_path = _first_existing([os.path.join(MODELS_DIR, n) for n in CAT_CANDIDATES])

        bin_meta_path = os.path.join(MODELS_DIR, BIN_META)
        cat_meta_path = os.path.join(MODELS_DIR, CAT_META)

        # Yükleme
        BIN = joblib.load(bin_path)
        CAT = joblib.load(cat_path)

        BIN_META_OBJ = joblib.load(bin_meta_path)
        CAT_META_OBJ = joblib.load(cat_meta_path)

        # Meta alanları
        BIN_COLS = list(BIN_META_OBJ.get("columns", []))
        CAT_COLS = list(CAT_META_OBJ.get("columns", []))
        CAT_CLASSES = list(CAT_META_OBJ.get("classes", []))

        if not BIN_COLS or not CAT_COLS or not CAT_CLASSES:
            raise ValueError("Meta dosyalarında beklenen anahtarlar yok: columns/classes")

        return BIN, CAT, BIN_COLS, CAT_COLS, CAT_CLASSES

    except Exception as e:
        raise RuntimeError(
            "Modeller yüklenemedi. 'models/' dizininde aşağıdakilerin olduğundan emin olun:\n"
            f" - {BIN_NAME}\n"
            f" - {BIN_META}\n"
            f" - ({' veya '.join(CAT_CANDIDATES)})\n"
            f" - {CAT_META}\n"
            f"Orijinal hata: {e}"
        )


# =========================
# 2) Yardımcılar (tip/sızıntı/hizalama)
# =========================
def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Temel tip/kısıt düzeltmeleri."""
    out = df.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str)
    if "event_hour" in out.columns:
        out["event_hour"] = pd.to_numeric(out["event_hour"], errors="coerce").fillna(-1).astype(int)
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    return out

def _filter_history_t_minus_1(df: pd.DataFrame, when: datetime) -> pd.DataFrame:
    """
    t−1 kuralı: 'when' tarihinden SONRAKİ satırları kesinlikle dışla.
    Varsayım: 'date' gün seviyesinde (YYYY-MM-DD), saat için 'event_hour' var.
    """
    if "date" not in df.columns:
        return df
    cutoff_day = when.date().isoformat()
    return df[df["date"] <= cutoff_day].copy()

def _last_row_for_hour(df: pd.DataFrame, geoid: str, hour: int) -> Optional[pd.Series]:
    """Aynı GEOID & event_hour için en yakın geçmiş satır; yoksa GEOID’in en son satırı."""
    g = df[df["GEOID"].astype(str) == str(geoid)]
    if g.empty:
        return None
    if "event_hour" in g.columns:
        gh = g[g["event_hour"] == int(hour)]
        if not gh.empty:
            return gh.sort_values("date").iloc[-1]
    return g.sort_values("date").iloc[-1]

def _row_to_aligned_frames(row: pd.Series, bin_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Eğitimde kullanılan kolon isimlerine göre tek-satır X_bin ve X_cat üretir (eksikler NaN)."""
    xb = pd.DataFrame([{c: (row[c] if c in row.index else np.nan) for c in bin_cols}])
    xc = pd.DataFrame([{c: (row[c] if c in row.index else np.nan) for c in cat_cols}])

    # Tip düzeltmeleri (kritik alanlar)
    if "event_hour" in xb.columns:
        xb["event_hour"] = pd.to_numeric(xb["event_hour"], errors="coerce").fillna(-1).astype(int)
    if "event_hour" in xc.columns:
        xc["event_hour"] = pd.to_numeric(xc["event_hour"], errors="coerce").fillna(-1).astype(int)
    if "GEOID" in xb.columns:
        xb["GEOID"] = xb["GEOID"].astype(str)
    if "GEOID" in xc.columns:
        xc["GEOID"] = xc["GEOID"].astype(str)
    return xb, xc


# =========================
# 3) Feature build + Tahmin
# =========================
def build_features_point(history_df: pd.DataFrame, geoid: str, when: datetime) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    'history_df' = t−1'e kadar olan veri (sızıntısız).
      1) when gününden SONRAKİ satırları at
      2) aynı GEOID & event_hour için en son satır; yoksa GEOID için en son satır
      3) eğitim kolonlarıyla hizalı tek satırlık X_bin ve X_cat döndür
    """
    _, _, BIN_COLS, CAT_COLS, _ = _load_models()  # sadece kolon adları için
    h = _ensure_types(history_df)
    h = _filter_history_t_minus_1(h, when)

    s = _last_row_for_hour(h, geoid, when.hour)
    if s is None:
        raise RuntimeError(f"Geçmişte satır bulunamadı: GEOID={geoid}")

    xb, xc = _row_to_aligned_frames(s, BIN_COLS, CAT_COLS)
    return xb, xc

def predict_point(history_df: pd.DataFrame, geoid: str, when: datetime) -> Dict:
    """
    Tek GEOID & zaman için:
      - p_occ  = P(Y=1) (ikili model)
      - top3   = [(kategori, P(cat ∧ olay)), ...]
    """
    BIN, CAT, _, _, CAT_CLASSES = _load_models()
    xb, xc = build_features_point(history_df, geoid, when)

    # Olasılıklar
    p_occ = float(BIN.predict_proba(xb)[0, 1])             # P(Y=1)
    p_cat = np.array(CAT.predict_proba(xc)[0], dtype=float)  # P(cat | Y=1) gibi düşünülür

    # Joint: P(cat ∧ olay) ≈ P(Y=1) * P(cat | Y=1)
    top = sorted([(str(CAT_CLASSES[i]), p_occ * p_cat[i]) for i in range(len(CAT_CLASSES))],
                 key=lambda x: x[1], reverse=True)

    return {
        "geoid": str(geoid),
        "when": when.isoformat(),
        "p_occ": p_occ,
        "top3": top[:3],
    }

def predict_many(history_df: pd.DataFrame, geoids: List[str], when: datetime) -> pd.DataFrame:
    """
    Çoklu GEOID tahmini (hata toleranslı).
    Dönen kolonlar: GEOID, when, p_occ, top_cat, top_cat_p
    """
    rows = []
    for g in geoids:
        try:
            r = predict_point(history_df, g, when)
            if r["top3"]:
                k1, p1 = r["top3"][0]
            else:
                k1, p1 = None, None
            rows.append({
                "GEOID": str(g),
                "when": r["when"],
                "p_occ": r["p_occ"],
                "top_cat": k1,
                "top_cat_p": p1,
            })
        except Exception:
            rows.append({
                "GEOID": str(g),
                "when": when.isoformat(),
                "p_occ": np.nan,
                "top_cat": None,
                "top_cat_p": None,
            })
    return pd.DataFrame(rows)
