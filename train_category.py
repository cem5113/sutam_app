# train_category.py — Multiclass (XGBoost) — FULL REVIZE
#!/usr/bin/env python3
from __future__ import annotations
import argparse, joblib, pandas as pd, numpy as np
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline as skl_makepipe
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from xgboost import XGBClassifier

FEATS = [
    "GEOID","event_hour","day_of_week_x","month_x","season_x",
    "is_holiday","is_weekend","is_night","is_school_hour","is_business_hour",
    "poi_risk_score","poi_total_count","bus_stop_count","train_stop_count",
    "distance_to_police","is_near_police","is_near_government",
    "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
    "911_request_count_daily(before_24_hours)","311_request_count",
    "wx_tavg","wx_prcp"
]

def load_df(path: str, rare_threshold: int = 50):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)

    # sadece olay olan satırlar
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    df = df[df["Y_label"] == 1].copy()

    need = ["GEOID","date","event_hour","category","latitude","longitude"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise SystemExit(f"Eksik kolonlar: {miss}")

    df["GEOID"] = df["GEOID"].astype(str)
    df["date"] = df["date"].astype(str)
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")

    keep = [c for c in FEATS if c in df.columns]
    if not keep:
        raise SystemExit("Feature kolonları yok (FEATS hepsi eksik).")

    X = df[keep].copy()
    y = df["category"].astype(str).to_numpy()

    # sınıf birleştirme (nadirler -> "Other")
    vc = pd.Series(y).value_counts()
    rare = set(vc[vc < rare_threshold].index)
    if rare:
        y = np.array([lbl if lbl not in rare else "Other" for lbl in y])

    # NaN/∞ temizlik + düşük varyans filtre
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X.drop(columns=all_nan, inplace=True)

    low_var = []
    for c in X.columns:
        vc = X[c].value_counts(dropna=False, normalize=True)
        if vc.shape[0] <= 1 or (vc.iloc[0] >= 0.995):  # %99.5 tek değer
            low_var.append(c)
    if low_var:
        X.drop(columns=low_var, inplace=True)

    return X, y

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID"); cat_cols.append("GEOID")

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", ohe)])

    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    # XGBoost multiclass — stabil ve hızlı
    xgb = XGBClassifier(
        objective="multi:softprob",
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("clf", xgb)])

def main(args):
    X, y = load_df(args.input, rare_threshold=args.rare_threshold)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(X)
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    print(classification_report(y_te, y_pred, digits=4))

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, f"{args.outdir}/sutam_category_xgb.joblib")
    meta = {"columns": X.columns.tolist(),
            "classes": sorted(pd.Series(y).unique().tolist())}
    joblib.dump(meta, f"{args.outdir}/category_meta.pkl")

    print(f"✅ Kaydedildi: {args.outdir}/sutam_category_xgb.joblib, {args.outdir}/category_meta.pkl")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--rare-threshold", type=int, default=50,
                    help="Bu sayıdan az örneği olan sınıfları 'Other' yapar.")
    main(ap.parse_args())
