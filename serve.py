from __future__ import annotations
import joblib, pandas as pd, numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Modelleri yükle
BIN = joblib.load("models/sutam_binary_stack_cal.joblib")
CAT = joblib.load("models/sutam_category_lgbm.joblib")
BIN_META = joblib.load("models/binary_meta.pkl")
CAT_META = joblib.load("models/category_meta.pkl")

BIN_COLS = BIN_META["columns"]
CAT_COLS = CAT_META["columns"]
CAT_CLASSES = CAT_META["classes"]

def _fallback(val, default=0):
    try: return float(val)
    except: return default

def build_features_point(history_df: pd.DataFrame, geoid: str, when: datetime) -> pd.DataFrame:
    """
    'history_df' = t-1'e kadar olan veri (sızıntısız).
    Basit strateji: aynı GEOID ve aynı 'event_hour' için en yakın geçmiş satırı al,
    gerekli sayısal alanların son değerlerini/rolling özetlerini kullan.
    İstersen burada kısa-ufuk forecast ekleyebilirsin.
    """
    h = history_df.copy()
    h = h[h["GEOID"].astype(str)==str(geoid)].copy()
    h["event_hour"] = pd.to_numeric(h["event_hour"], errors="coerce").fillna(0).astype(int)
    target_hour = when.hour
    cand = h[h["event_hour"]==target_hour].sort_values("date")
    if cand.empty:
        cand = h.sort_values("date")
    row = cand.iloc[-1:]  # en son bilinen

    # Çıkarma: yalnızca eğitimde kullanılan kolonlar
    xb = pd.DataFrame([{c: row[c].iloc[0] if c in row.columns else np.nan for c in BIN_COLS}])
    xc = pd.DataFrame([{c: row[c].iloc[0] if c in row.columns else np.nan for c in CAT_COLS}])
    return xb, xc

def predict_point(history_df: pd.DataFrame, geoid: str, when: datetime) -> Dict:
    xb, xc = build_features_point(history_df, geoid, when)
    p_occ = float(BIN.predict_proba(xb)[0,1])
    p_cat = CAT.predict_proba(xc)[0]  # sınıf olasılıkları
    top = sorted([(CAT_CLASSES[i], p_occ * float(p_cat[i])) for i in range(len(CAT_CLASSES))],
                 key=lambda x: x[1], reverse=True)
    return {
        "geoid": geoid,
        "when": when.isoformat(),
        "p_occ": p_occ,
        "top3": top[:3],  # (kategori, joint p)
    }

def predict_many(history_df: pd.DataFrame, geoids: List[str], when: datetime) -> pd.DataFrame:
    rows = []
    for g in geoids:
        r = predict_point(history_df, g, when)
        rows.append({"GEOID": g, "when": r["when"], "p_occ": r["p_occ"],
                     "k1": r["top3"][0][0] if r["top3"] else None,
                     "p1": r["top3"][0][1] if r["top3"] else None})
    return pd.DataFrame(rows)
