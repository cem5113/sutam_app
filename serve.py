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
BIN_META = "binary_meta.pkl"  # {columns: List[str], ...}

# Kategori için sıralı arama (hangisi varsa onu yükle)
CAT_CANDIDATES = [
    "sutam_category_xgb.joblib",   # önerilen
    "sutam_category_lgbm.joblib",  # alternatif
]
CAT_META = "category_meta.pkl"  # {columns: List[str], classes: List[str]}

# =========================
# 1) Model yükleme (güvenli)
# =========================

def _first_existing(paths: List[str]) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(" / ".join(paths))


def _safe_joblib_load(path: str):
    """Joblib yüklemesini güvenli/uyumlu moda al."""
    try:
        return joblib.load(path, mmap_mode="r")
    except Exception:
        # Eski joblib sürümleri veya farklı protokoller için fallback
        return joblib.load(path)


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
        BIN = _safe_joblib_load(bin_path)
        CAT = _safe_joblib_load(cat_path)

        BIN_META_OBJ = _safe_joblib_load(bin_meta_path)
        CAT_META_OBJ = _safe_joblib_load(cat_meta_path)

        # Meta alanları
        BIN_COLS = list(BIN_META_OBJ.get("columns", []))
        CAT_COLS = list(CAT_META_OBJ.get("columns", []))
        CAT_CLASSES = list(CAT_META_OBJ.get("classes", []))

        if not BIN_COLS:
            raise ValueError("Binary meta: 'columns' boş.")
        if not CAT_COLS:
            raise ValueError("Category meta: 'columns' boş.")
        if not CAT_CLASSES:
            raise ValueError("Category meta: 'classes' boş.")

        # predict_proba güvenlik kontrolü
        for est, name in ((BIN, "binary"), (CAT, "category")):
            if not hasattr(est, "predict_proba"):
                raise TypeError(f"{name} modelinde predict_proba yok.")

        return BIN, CAT, BIN_COLS, CAT_COLS, CAT_CLASSES

    except Exception as e:
        try:
            listing = os.listdir(MODELS_DIR)
        except Exception:
            listing = ["<erişim yok>"]
        raise RuntimeError(
            "Modeller yüklenemedi. 'models/' dizininde aşağıdakilerin olduğundan emin olun:\n"
            f" - {BIN_NAME}\n"
            f" - {BIN_META}\n"
            f" - ({' veya '.join(CAT_CANDIDATES)})\n"
            f" - {CAT_META}\n"
            f"Bulunan dosyalar: {listing}\n"
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
    # Muhtemel kopya kolonlar
    out = out.loc[:, ~out.columns.duplicated()].copy()
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
# 3) Feature build + Tahmin (tek / çoklu)
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
    p_cat = np.array(CAT.predict_proba(xc)[0], dtype=float)  # P(cat | Y=1) varsayımı

    # Joint: P(cat ∧ olay) ≈ P(Y=1) * P(cat | Y=1)
    top = sorted([(str(CAT_CLASSES[i]), p_occ * p_cat[i]) for i in range(len(CAT_CLASSES))],
                 key=lambda x: x[1], reverse=True)

    return {
        "geoid": str(geoid),
        "when": when.isoformat(),
        "p_occ": p_occ,
        "top3": top[:3],
    }


def _choose_rows_for_many(df: pd.DataFrame, geoids: List[str], when: datetime) -> pd.DataFrame:
    """Çoklu GEOID için, her GEOID adına kullanılacak tek bir geçmiş satırı seç."""
    h = _ensure_types(df)
    h = _filter_history_t_minus_1(h, when)

    # İlgili GEOID'lere indir
    gset = set(map(str, geoids))
    h = h[h["GEOID"].astype(str).isin(gset)]
    if h.empty:
        return h.iloc[:0]

    # Önce aynı saat (event_hour == when.hour) için en yeni satır, yoksa GEOID bazında en yeni satır
    if "event_hour" in h.columns:
        # Saat eşleşenleri al
        same_hour = h[h["event_hour"] == int(when.hour)]
        same_hour = same_hour.sort_values(["GEOID", "date"]).groupby("GEOID", as_index=False).tail(1)

        # Eşleşmeyenler için GEOID bazında en yeni satır
        newest_any = (
            h.sort_values(["GEOID", "date"]).groupby("GEOID", as_index=False).tail(1)
        )
        # Birleşim: öncelik same_hour
        chosen = pd.concat([same_hour, newest_any], ignore_index=True)
        chosen = chosen.sort_values(["GEOID", "event_hour"], na_position="last").drop_duplicates("GEOID", keep="first")
    else:
        chosen = h.sort_values(["GEOID", "date"]).groupby("GEOID", as_index=False).tail(1)

    return chosen


def predict_many(history_df: pd.DataFrame, geoids: List[str], when: datetime) -> pd.DataFrame:
    """
    Çoklu GEOID tahmini (vektörleştirilmiş seçim, hata toleranslı).
    Dönen kolonlar: GEOID, when, p_occ, top_cat, top_cat_p
    """
    BIN, CAT, BIN_COLS, CAT_COLS, CAT_CLASSES = _load_models()

    chosen = _choose_rows_for_many(history_df, geoids, when)
    if chosen.empty:
        # İstenen GEOID'ler için NaN döndür
        return pd.DataFrame({
            "GEOID": list(map(str, geoids)),
            "when": when.isoformat(),
            "p_occ": np.nan,
            "top_cat": None,
            "top_cat_p": None,
        })

    # Eğitim kolonlarına hizala
    Xb = pd.DataFrame([{c: (row.get(c, np.nan)) for c in BIN_COLS} for _, row in chosen.iterrows()])
    Xc = pd.DataFrame([{c: (row.get(c, np.nan)) for c in CAT_COLS} for _, row in chosen.iterrows()])

    # Tip güvenliği
    for X in (Xb, Xc):
        if "event_hour" in X.columns:
            X["event_hour"] = pd.to_numeric(X["event_hour"], errors="coerce").fillna(-1).astype(int)
        if "GEOID" in X.columns:
            X["GEOID"] = X["GEOID"].astype(str)

    # Olasılıklar (toplu)
    p_occ = BIN.predict_proba(Xb)[:, 1].astype(float)
    p_cat = CAT.predict_proba(Xc).astype(float)  # shape: (n, n_classes)

    # Joint olasılık ve top-1 sınıf
    joint = p_cat * p_occ.reshape(-1, 1)  # (n, n_classes)
    idx = np.argmax(joint, axis=1)

    out = pd.DataFrame({
        "GEOID": chosen["GEOID"].astype(str).values,
        "when": when.isoformat(),
        "p_occ": p_occ,
        "top_cat": [str(CAT_CLASSES[i]) for i in idx],
        "top_cat_p": joint[np.arange(len(idx)), idx],
    })

    # İstenen sıralama ile uyumlu hale getir (geoids sırası)
    order = {str(g): i for i, g in enumerate(geoids)}
    out["_ord"] = out["GEOID"].map(order)
    out = out.sort_values(["_ord", "GEOID"]).drop(columns="_ord")

    # Eksik kalan GEOID'ler için NaN satır ekle
    missing = [g for g in map(str, geoids) if g not in set(out["GEOID"]) ]
    if missing:
        out = pd.concat([
            out,
            pd.DataFrame({
                "GEOID": missing,
                "when": when.isoformat(),
                "p_occ": np.nan,
                "top_cat": None,
                "top_cat_p": None,
            })
        ], ignore_index=True).sort_values(["GEOID"]).reset_index(drop=True)

    return out
