#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_df(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    # minimal zorunlu kolonlar:
    need = ["GEOID","date","event_hour","Y_label","latitude","longitude"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise SystemExit(f"Eksik kolonlar: {miss}")
    # tipler
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    # inference'ta kullanılabilir örnek özellik seti (yalın ama güçlü)
    keep = [c for c in [
        "GEOID","event_hour","day_of_week_x","month_x","season_x",
        "is_holiday","is_weekend","is_night","is_school_hour","is_business_hour",
        "poi_risk_score","poi_total_count","bus_stop_count","train_stop_count",
        "distance_to_police","is_near_police","is_near_government",
        "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
        "911_request_count_daily(before_24_hours)","311_request_count",
        "wx_tavg","wx_prcp"
    ] if c in df.columns]
    X = df[keep].copy()
    y = df["Y_label"].values
    # train/test zaman ayrımı (basit kural: son 20% test)
    # İstersen argümanla sabit tarih de verilebilir
    df["_t"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = df["_t"].quantile(0.80)
    tr_idx = df["_t"] < cutoff
    te_idx = df["_t"] >= cutoff
    return X, y, tr_idx.values, te_idx.values

def make_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    # GEOID'i kategoriye al (target encoding kullanmak istersen burada değiştirirsin)
    if "GEOID" in num_cols:
        num_cols.remove("GEOID"); cat_cols.append("GEOID")

    pre = ColumnTransformer([
        ("num","passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
    ])

    base = [
        ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=-1,
                                subsample=0.8, colsample_bytree=0.8, class_weight="balanced")),
        ("xgb",  XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                               eval_metric="logloss", scale_pos_weight=1.0))
    ]
    meta = LogisticRegression(max_iter=500, class_weight="balanced")
    stack = StackingClassifier(estimators=base, final_estimator=meta, passthrough=True, n_jobs=-1)
    clf = Pipeline([("pre", pre), ("stack", stack)])
    return clf

def main(args):
    X, y, tr, te = load_df(args.input)
    clf = make_pipeline(X)
    clf.fit(X.iloc[tr], y[tr])
    # Kalibrasyon (Platt)
    cal = CalibratedClassifierCV(clf, method="sigmoid", cv="prefit")
    cal.fit(X.iloc[tr], y[tr])
    p = cal.predict_proba(X.iloc[te])[:,1]
    ap = average_precision_score(y[te], p)
    print(f"[Binary] Test AP (PR-AUC): {ap:.4f}")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(cal, f"{args.outdir}/sutam_binary_stack_cal.joblib")
    joblib.dump({"columns": X.columns.tolist()}, f"{args.outdir}/binary_meta.pkl")
    print("✅ Kaydedildi: sutam_binary_stack_cal.joblib")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="fr_crime_09.parquet (veya CSV)")
    ap.add_argument("--outdir", default="models")
    main(ap.parse_args())
