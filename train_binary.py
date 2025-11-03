#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_binary.py — SUTAM ikili (Y_label) modeli
- Robust veri yükleme ve zaman-bazlı (leakage-safe) split
- Stacking (XGB + ExtraTrees) + L2-logistic meta
- Sınıf dengesizliği için otomatik scale_pos_weight ve class_weight
- OHE uyumluluğu (sklearn >=1.2 ve <1.2 için)
- Kalibrasyon (CV tabanlı) + metriklerin kaydı
- Tek bir .joblib çıktısı içinde tüm pipeline + kalibratör
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline as skl_makepipe
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.ensemble import StackingClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

# ========= Konfig =========
REQ_COLS = ["GEOID", "date", "event_hour", "Y_label", "latitude", "longitude"]
FEATS = [
    "GEOID", "event_hour", "day_of_week_x", "month_x", "season_x",
    "is_holiday", "is_weekend", "is_night", "is_school_hour", "is_business_hour",
    "poi_risk_score", "poi_total_count", "bus_stop_count", "train_stop_count",
    "distance_to_police", "is_near_police", "is_near_government",
    "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d",
    "911_request_count_daily(before_24_hours)", "311_request_count",
    "wx_tavg", "wx_prcp",
]

@dataclass
class TrainMeta:
    columns: list
    cutoff_date: str
    pos_frac_train: float
    n_features: int
    notes: str = "time-based split @80% quantile; leakage-safe by day"


# ========= Yardımcılar =========
def _onehot_encoder():
    try:
        # sklearn >=1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # eski sürümler
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def _auto_pos_weight(y: np.ndarray) -> float:
    # scale_pos_weight = Neg/Pos (XGBoost için klasik oran)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return max(n_neg / n_pos, 1.0) if n_pos > 0 else 1.0


# ========= Veri Yükleme & Split =========
def load_df(path: str):
    df = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)

    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"Eksik kolonlar: {miss}")

    # tip güvenliği
    df["GEOID"] = df["GEOID"].astype(str)
    df["date"] = df["date"].astype(str)
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce")
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")

    if df["event_hour"].isna().any():
        df["event_hour"] = df["event_hour"].fillna(df["event_hour"].mode(dropna=True).iloc[0])
    df["event_hour"] = df["event_hour"].astype("int16")

    keep = [c for c in FEATS if c in df.columns]
    if not keep:
        raise SystemExit("Feature kolonları bulunamadı (keep listesi boş).")

    X = df[keep].copy()
    y = df["Y_label"].to_numpy()

    # ekstrem değer temizlik
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X.drop(columns=all_nan_cols, inplace=True)

    # zaman tabanlı bölme (gün seviyesinde; leakage-safe)
    df["_t"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = df["_t"].quantile(0.80)
    tr_mask = (df["_t"] < cutoff).to_numpy()
    te_mask = (df["_t"] >= cutoff).to_numpy()

    return X, y, tr_mask, te_mask, cutoff


# ========= Pipeline =========
def build_pipeline(X: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID")
        cat_cols.append("GEOID")

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", _onehot_encoder()),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Sınıf dengesizliği: pos_weight
    spw = _auto_pos_weight(y_train)

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    et = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    base = [("xgb", xgb), ("et", et)]

    final_est = skl_makepipe(
        SimpleImputer(strategy="median"),
        LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=seed),
    )

    cv_part = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    stack = StackingClassifier(
        estimators=base,
        final_estimator=final_est,
        passthrough=True,
        cv=cv_part,
        n_jobs=-1,
        stack_method="predict_proba",
    )

    return Pipeline([
        ("pre", pre),
        ("stack", stack),
    ])


# ========= Ana akış =========
def main(args):
    X, y, tr, te, cutoff = load_df(args.input)
    print(f"[INFO] X shape={X.shape}, y={y.shape}, train={tr.sum()} test={te.sum()} (cutoff={cutoff.date()})")

    pipe_uncal = build_pipeline(X, y_train=y[tr], seed=args.seed)

    # Kalibrasyon: CV ile (prefit değil)
    cal = CalibratedClassifierCV(pipe_uncal, method="sigmoid", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=args.seed))
    cal.fit(X.iloc[tr], y[tr])

    p = cal.predict_proba(X.iloc[te])[:, 1]
    ap = average_precision_score(y[te], p)
    print(f"[Binary] Test AP (PR-AUC): {ap:.4f}")

    # PR eğrisi ve metrikleri kaydet
    pr_p, pr_r, pr_th = precision_recall_curve(y[te], p)
    metrics = {
        "ap_pr_auc": float(ap),
        "cutoff_date": str(cutoff.date()),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "pos_frac_train": float(y[tr].mean()),
        "pos_frac_test": float(y[te].mean()),
        "seed": int(args.seed),
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    joblib.dump(cal, outdir / "sutam_binary_stack_cal.joblib")
    meta = TrainMeta(columns=X.columns.tolist(), cutoff_date=str(cutoff.date()), pos_frac_train=float(y[tr].mean()), n_features=X.shape[1])
    joblib.dump({"columns": meta.columns, "cutoff_date": meta.cutoff_date, "pos_frac_train": meta.pos_frac_train}, outdir / "binary_meta.pkl")

    # CSV/JSON raporları
    pd.DataFrame({"precision": pr_p, "recall": pr_r}).to_csv(outdir / "pr_curve_binary.csv", index=False)
    pd.Series(metrics).to_json(outdir / "metrics_binary.json", indent=2)

    print("✅ Kaydedildi:")
    print(" -", outdir / "sutam_binary_stack_cal.joblib")
    print(" -", outdir / "binary_meta.pkl")
    print(" -", outdir / "pr_curve_binary.csv")
    print(" -", outdir / "metrics_binary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fr_crime_09.parquet (veya CSV)")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
