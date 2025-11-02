#!/usr/bin/env python3
from __future__ import annotations

import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline as skl_makepipe
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.ensemble import StackingClassifier
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


REQ_COLS = ["GEOID", "date", "event_hour", "Y_label", "latitude", "longitude"]
FEATS = [
    "GEOID","event_hour","day_of_week_x","month_x","season_x",
    "is_holiday","is_weekend","is_night","is_school_hour","is_business_hour",
    "poi_risk_score","poi_total_count","bus_stop_count","train_stop_count",
    "distance_to_police","is_near_police","is_near_government",
    "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
    "911_request_count_daily(before_24_hours)","311_request_count",
    "wx_tavg","wx_prcp"
]


def load_df(path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    # Zorunlu kolon kontrolü
    miss = [c for c in REQ_COLS if c not in df.columns]
    if miss:
        raise SystemExit(f"Eksik kolonlar: {miss}")

    # Tipler
    df["GEOID"] = df["GEOID"].astype(str)
    df["date"] = df["date"].astype(str)
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce")
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")

    if df["event_hour"].isna().any():
        # çok az NaN varsa saat moduyla doldur
        df["event_hour"] = df["event_hour"].fillna(df["event_hour"].mode(dropna=True).iloc[0])
    df["event_hour"] = df["event_hour"].astype("int16")

    # Feature seçimi (mevcutta olanlar)
    keep = [c for c in FEATS if c in df.columns]
    if not keep:
        raise SystemExit("Feature kolonları bulunamadı (keep listesi boş).")

    X = df[keep].copy()
    y = df["Y_label"].to_numpy()

    # Zaman bazlı dış holdout: son %20 test
    df["_t"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = df["_t"].quantile(0.80)
    tr_idx = (df["_t"] < cutoff).to_numpy()
    te_idx = (df["_t"] >= cutoff).to_numpy()

    # Temizlik: inf -> NaN, tamamen NaN olan kolonları at
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan_cols = [c for c in X.columns if X[c].isna().all()]
    if all_nan_cols:
        X.drop(columns=all_nan_cols, inplace=True)

    return X, y, tr_idx, te_idx


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Numerik / kategorik ayrımı
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID")
        cat_cols.append("GEOID")

    # Önişleme (NaN güvenli)
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])

    try:
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse=True)

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", cat_enc),
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Baz modeller
    lgbm = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    base = [("lgbm", lgbm), ("xgb", xgb)]

    # Meta (final) estimator: imputer'lı LR (NaN güvenli)
    final_est = skl_makepipe(
        SimpleImputer(strategy="median"),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42),
    )

    # İç CV: partition garantili (Stacking için gerekli)
    cv_part = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    stack = StackingClassifier(
        estimators=base,
        final_estimator=final_est,
        passthrough=True,
        cv=cv_part,
        n_jobs=-1,
        stack_method="predict_proba",
    )

    return Pipeline([("pre", pre), ("stack", stack)])


def main(args):
    X, y, tr, te = load_df(args.input)
    print(f"[INFO] X shape={X.shape}, y={y.shape}, train={tr.sum()} test={te.sum()}")

    clf = build_pipeline(X)

    # Fit (train dilimi)
    clf.fit(X.iloc[tr], y[tr])

    # Kalibrasyon (Platt) — prefit
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    cal.fit(X.iloc[tr], y[tr])

    # Değerlendirme
    p = cal.predict_proba(X.iloc[te])[:, 1]
    ap = average_precision_score(y[te], p)
    print(f"[Binary] Test AP (PR-AUC): {ap:.4f}")

    # Kayıt
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, f"{args.outdir}/sutam_binary_stack_cal.joblib")
    joblib.dump({"columns": X.columns.tolist()}, f"{args.outdir}/binary_meta.pkl")
    print(f"✅ Kaydedildi: {args.outdir}/sutam_binary_stack_cal.joblib, {args.outdir}/binary_meta.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fr_crime_09.parquet (veya CSV)")
    ap.add_argument("--outdir", default="models")
    main(ap.parse_args())
