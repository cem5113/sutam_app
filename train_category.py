#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

def load_df(path: str):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    df = df[df["Y_label"] == 1].copy()  # sadece olay olanlar
    need = ["category","GEOID","event_hour","date"]
    for c in need:
        if c not in df.columns: raise SystemExit(f"eksik: {c}")
    # Özellikler: binary ile aynı set (subcategory kullanmıyoruz)
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
    y = df["category"].astype("category")
    # zaman ayrımı
    df["_t"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = df["_t"].quantile(0.80)
    tr = df["_t"] < cutoff
    te = df["_t"] >= cutoff
    return X, y, tr.values, te.values

def make_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID"); cat_cols.append("GEOID")
    pre = ColumnTransformer([
        ("num","passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
    ])
    lgbm = LGBMClassifier(objective="multiclass", n_estimators=400, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8)
    return Pipeline([("pre", pre), ("lgbm", lgbm)])

def main(args):
    X, y, tr, te = load_df(args.input)
    pipe = make_pipeline(X)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    yhat = pipe.predict(X.iloc[te])
    f1 = f1_score(y.iloc[te], yhat, average="macro")
    print(f"[Category] Test Macro-F1: {f1:.3f}")
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, f"{args.outdir}/sutam_category_lgbm.joblib")
    joblib.dump({"columns": X.columns.tolist(), "classes": y.cat.categories.tolist()},
                f"{args.outdir}/category_meta.pkl")
    print("✅ Kaydedildi: sutam_category_lgbm.joblib")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="models")
    main(ap.parse_args())
