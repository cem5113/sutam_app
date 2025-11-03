#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_category.py — SUTAM çok sınıflı kategori modeli (FULL REVIZE)
- Girdi: fr_crime_09.parquet (veya CSV)
- Yalnızca Y_label==1 (olay) satırları ile kategori sınıflandırması
- Nadir sınıfları 'Other' altında birleştirme (eşik: --rare-threshold)
- Zaman-bazlı ayrım (leakage-safe): son %20 test
- OHE uyumluluğu (sklearn >=1.2 ve <1.2)
- XGBoost (multi:softprob) + sınıf dengesizliği için örnek ağırlıkları
- Metrikler: macro F1, accuracy@1, accuracy@3, logloss; sınıf raporu
- Çıktılar: sutam_category_xgb.joblib, category_meta.pkl, metrics_category.json, report_category.csv
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline as skl_makepipe
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, log_loss, classification_report

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


def _onehot():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def load_df(path: str, rare_threshold: int = 50):
    df = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)

    # yalnız olaylar
    df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    df = df[df["Y_label"] == 1].copy()

    need = ["GEOID","date","event_hour","category","latitude","longitude"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"Eksik kolonlar: {miss}")

    df["GEOID"] = df["GEOID"].astype(str)
    df["date"] = df["date"].astype(str)
    df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")

    keep = [c for c in FEATS if c in df.columns]
    if not keep:
        raise SystemExit("Feature kolonları yok (FEATS hepsi eksik).")

    X = df[keep].copy()
    y = df["category"].astype(str).to_numpy()

    # nadirleri Other'a al
    vc = pd.Series(y).value_counts()
    rare = set(vc[vc < rare_threshold].index)
    if rare:
        y = np.array([lbl if lbl not in rare else "Other" for lbl in y])

    # temizlik
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_nan = [c for c in X.columns if X[c].isna().all()]
    if all_nan:
        X.drop(columns=all_nan, inplace=True)

    # zaman-bazlı bölme (son %20 test)
    t = pd.to_datetime(df["date"], errors="coerce")
    cutoff = t.quantile(0.80)
    tr_mask = (t < cutoff).to_numpy()
    te_mask = (t >= cutoff).to_numpy()

    return X, y, tr_mask, te_mask, cutoff


def build_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if "GEOID" in num_cols:
        num_cols.remove("GEOID")
        cat_cols.append("GEOID")

    num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", _onehot())])

    pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

    xgb = XGBClassifier(
        objective="multi:softprob",
        n_estimators=700,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline([("pre", pre), ("clf", xgb)])


def _class_weights(y: np.ndarray) -> dict:
    # sınıf frekansına göre ters orantılı ağırlık
    vc = pd.Series(y).value_counts()
    total = float(len(y))
    weights = {cls: total / (len(vc) * float(cnt)) for cls, cnt in vc.items()}
    return weights


def _topk_acc(y_true: np.ndarray, proba: np.ndarray, classes: list[str], k: int = 3) -> float:
    cls_index = {c: i for i, c in enumerate(classes)}
    true_idx = np.array([cls_index[c] for c in y_true])
    topk = np.argsort(proba, axis=1)[:, -k:]
    hits = (topk == true_idx.reshape(-1, 1)).any(axis=1)
    return float(hits.mean())


def main(args):
    X, y, tr, te, cutoff = load_df(args.input, rare_threshold=args.rare_threshold)
    print(f"[INFO] X={X.shape}, y={y.shape}, train={tr.sum()} test={te.sum()} (cutoff={cutoff.date()})")

    pipe = build_pipeline(X, random_state=args.seed)

    # sınıf ağırlığı → örnek ağırlığına çevir (XGBoost çok-sınıflı için)
    wmap = _class_weights(y[tr])
    sample_weight = np.array([wmap[c] for c in y[tr]])

    pipe.fit(X.iloc[tr], y[tr], clf__sample_weight=sample_weight)

    # Tahmin/olasılık
    y_hat = pipe.predict(X.iloc[te])
    y_proba = pipe.predict_proba(X.iloc[te])
    classes = list(pipe.named_steps["clf"].classes_)

    # Metrikler
    f1_macro = f1_score(y[te], y_hat, average="macro")
    acc1 = float((y_hat == y[te]).mean())
    acc3 = _topk_acc(y[te], y_proba, classes, k=3) if y_proba.shape[1] >= 3 else acc1
    ll = log_loss(y[te], y_proba, labels=classes)

    print(f"[Category] macro F1={f1_macro:.4f} | acc@1={acc1:.4f} | acc@3={acc3:.4f} | logloss={ll:.4f}")
    report_df = pd.DataFrame.from_dict(
        json.loads(classification_report(y[te], y_hat, output_dict=True)),
        orient="index"
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Model
    joblib.dump(pipe, outdir / "sutam_category_xgb.joblib")

    # Meta
    meta = {"columns": X.columns.tolist(), "classes": classes, "cutoff_date": str(cutoff.date())}
    joblib.dump(meta, outdir / "category_meta.pkl")

    # Raporlar
    metrics = {
        "f1_macro": float(f1_macro),
        "acc_top1": float(acc1),
        "acc_top3": float(acc3),
        "logloss": float(ll),
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "pos_frac": 1.0,  # yalnız Y=1 altkümesi kullanıldı
        "cutoff_date": str(cutoff.date()),
        "seed": int(args.seed),
        "rare_threshold": int(args.rare_threshold),
    }
    pd.Series(metrics).to_json(outdir / "metrics_category.json", indent=2)
    report_df.to_csv(outdir / "report_category.csv")

    print("✅ Kaydedildi:")
    print(" -", outdir / "sutam_category_xgb.joblib")
    print(" -", outdir / "category_meta.pkl")
    print(" -", outdir / "metrics_category.json")
    print(" -", outdir / "report_category.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--rare-threshold", type=int, default=50, help="Bu sayıdan az örneği olan sınıfları 'Other' yapar.")
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
