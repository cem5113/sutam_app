#!/usr/bin/env python3
# serve.py — SUTAM tahmin çekirdeği (stacking binary + multiclass)
from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, List, Tuple
from datetime import datetime

import joblib
import numpy as np
import pandas as pd


# =========================
# 1) Model yükleme (güvenli)
# =========================

MODELS_DIR = os.environ.get("SUTAM_MODELS_DIR", "models")

@lru_cache(maxsize=1)
def _load_models():
    """Modelleri ve meta bilgileri yükler; bulunamazsa açıklayıcı hata verir."""
    try:
        bin_path = os.path.join(MODELS_DIR, "sutam_binary_stack_cal.joblib")
        cat_path = os.path.join(MODELS_DIR, "sutam_category_lgbm.joblib")
        bin_meta = os.path.join(MODELS_DIR, "binary_meta.pkl")
        cat_meta = os.path.join(MODELS_DIR, "category_meta.pkl")

        BIN = joblib.load(bin_path)
        CAT = joblib.load(cat_path)
        BIN_META = joblib.load(bin_meta)
        CAT_META = joblib.load(cat_meta)

        BIN_COLS = list(BIN_META["columns"])
        CAT_COLS = list(CAT_META["columns"])
        CAT_CLASSES = list(CAT_META["classes"])
        return BIN, CAT, BIN_COLS, CAT_COLS, CAT_CLASSES
    except Exception as e:
        raise RuntimeError(
            "Modeller yüklenemedi. 'models/' dizininde aşağıdakilerin olduğundan emin olun:\n"
            " - sutam_binary_stack_cal.joblib\n"
            " - binary_meta.pkl\n"
            " - sutam_category_lgbm.joblib\n"
            " - category_meta.pkl\n"
            f"Orijinal hata: {e}"
        )


# =========================
# 2) Yardımcılar
# =========================

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    """Tahminden önce temel tip/kısıt düzeltmeleri."""
    out = df.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str)
    if "event_hour" in out.columns:
        out["event_hour"] = pd.to_numeric(out["event_hour"], errors="coerce").fillna(-1).astype(int)
    # date string olabilir; filtreleme için aynı formatı kullanacağız
    if "date" in out.columns:
        out["date"] = out["date"].astype(str)
    return out

def _filter_history_t_minus_1(df: pd.DataFrame, when: datetime) -> pd.DataFrame:
    """t−1 kuralı: 'when' tarih/saatinden SONRAKİ satırları kesinlikle kullanma."""
    if "date" not in df.columns:
        # Tarih yoksa olduğu gibi döndür (sızıntı korunamayabilir — veri modelleri öyleyse)
        return df
    # 'date' gün bazında tutuluyor diye varsayıyoruz (HH olmadan). Saat için event_hour var.
    # when.date()'e kadar olan satırları al (<= t-1 gün).
    cutoff_day = when.date().isoformat()
    return df[df["date"] <= cutoff_day].copy()

def _last_row_for_hour(df: pd.DataFrame, geoid: str, hour: int) -> pd.Series | None:
    """Aynı GEOID & hour için en yakın geçmiş satır; yoksa GEOID'in en son satırı."""
    g = df[df["GEOID"].astype(str) == str(geoid)]
    if g.empty:
        return None
    if "event_hour" in g.columns:
        gh = g[g["event_hour"] == int(hour)]
        if not gh.empty:
            return gh.sort_values("date").iloc[-1]
    # yedek: GEOID için en son satır
    return g.sort_values("date").iloc[-1]

def _row_to_aligned_frames(row: pd.Series, bin_cols: List[str], cat_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Eğitimde kullanılan kolon isimlerine göre tek satırlık DataFrame üretir (eksikler NaN)."""
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
    Strateji:
      1) 'when' gününden SONRAKİ satırları at
      2) aynı GEOID & event_hour için en son satır; yoksa GEOID için en son satır
      3) eğitim kolonlarıyla hizalanmış tek satırlık X_bin ve X_cat döndür
    """
    BIN, CAT, BIN_COLS, CAT_COLS, _ = _load_models()  # sadece kolon adları için
    h = _ensure_types(history_df)
    h = _filter_history_t_minus_1(h, when)

    s = _last_row_for_hour(h, geoid, when.hour)
    if s is None:
        raise RuntimeError(f"Geçmişte satır bulunamadı: GEOID={geoid}")

    xb, xc = _row_to_aligned_frames(s, BIN_COLS, CAT_COLS)
    return xb, xc

def predict_point(history_df: pd.DataFrame, geoid: str, when: datetime) -> Dict:
    """Tek GEOID & zaman için: P(Y=1) ve top-3 kategori joint olasılıkları."""
    BIN, CAT, _, _, CAT_CLASSES = _load_models()
    xb, xc = build_features_point(history_df, geoid, when)

    # Olasılıklar
    p_occ = float(BIN.predict_proba(xb)[0, 1])
    p_cat = CAT.predict_proba(xc)[0]  # multiclass dağılım

    # Joint: P(cat ∧ olay) = P(Y=1) * P(cat | Y=1)
    top = sorted([(str(CAT_CLASSES[i]), p_occ * float(p_cat[i])) for i in range(len(CAT_CLASSES))],
                 key=lambda x: x[1], reverse=True)

    return {
        "geoid": str(geoid),
        "when": when.isoformat(),
        "p_occ": p_occ,
        "top3": top[:3],  # (kategori, joint p)
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
