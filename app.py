# app.py — SUTAM • Gözlenen Risk Haritası (Y_label) + Tahmin (Stacking)
from __future__ import annotations

import os
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# ---- Modellerin yolu (serve.py bunu okur) ----
os.environ.setdefault("SUTAM_MODELS_DIR", "models")

# ---- Tahmin fonksiyonları (opsiyonel) ----
_PRED_POINT_OK = False
_PRED_MANY_OK  = False
try:
    from serve import predict_point
    _PRED_POINT_OK = True
except Exception:
    pass

try:
    from serve import predict_many
    _PRED_MANY_OK = True
except Exception:
    pass

# ---- Ayarlar ----
DATA_PATH  = os.getenv("SUTAM_DATA_PATH", "data/sf_crime_observed.csv")
PAGE_TITLE = "SUTAM — Gözlenen Risk Haritası (Y_label)"
MAP_INITIAL = {"lat": 37.7749, "lon": -122.4194, "zoom": 11.2}  # SF merkezi

st.set_page_config(page_title=PAGE_TITLE, layout="wide")


# =============== Veri Yükleme ===============
@st.cache_data(show_spinner=True)
def load_data_any(path: str) -> pd.DataFrame:
    """CSV veya Parquet yükler; mevcut faydalı kolonları alır ve tipleri düzeltir."""
    want_cols: List[str] = [
        "GEOID", "latitude", "longitude", "Y_label", "date", "event_hour",
        # tooltip için isteğe bağlı alanlar
        "category", "poi_risk_score", "911_request_count_daily(before_24_hours)",
        "neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d", "crime_count",
    ]
    try:
        if path.lower().endswith((".parquet", ".pq")):
            df_full = pd.read_parquet(path)
            usecols = [c for c in want_cols if c in df_full.columns]
            df = df_full[usecols].copy()
        else:
            head = pd.read_csv(path, nrows=1)
            usecols = [c for c in want_cols if c in head.columns]
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
    except Exception as e:
        st.error(f"Veri yüklenemedi: {path}\n{e}")
        return pd.DataFrame()

    # Tip düzeltmeleri
    if "Y_label" in df.columns:
        df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    if "event_hour" in df.columns:
        df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")
    if {"latitude", "longitude"}.issubset(df.columns):
        df = df.dropna(subset=["latitude", "longitude"])
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)

    # Tooltip güvenliği: eksik kolonları boş olarak ekle
    if "poi_risk_score" not in df.columns:
        df["poi_risk_score"] = np.nan
    if "911_request_count_daily(before_24_hours)" not in df.columns:
        df["911_request_count_daily(before_24_hours)"] = np.nan
    for c in ["neighbor_crime_24h", "neighbor_crime_72h", "neighbor_crime_7d", "crime_count"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


# =============== KPI ve Harita ===============
def kpi_block(df_sel: pd.DataFrame):
    total = int(len(df_sel))
    y1 = int(df_sel["Y_label"].sum()) if "Y_label" in df_sel.columns and total else 0
    rate = (100.0 * y1 / total) if total else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Satır (seçim)", f"{total:,}")
    c2.metric("Y=1 (olay)", f"{y1:,}")
    c3.metric("Y=1 oranı (%)", f"{rate:.2f}")

def make_map(df_sel: pd.DataFrame, show_y0: bool):
    """Y_label odaklı harita (gözlenen). Tooltip alanlarını güvenli hale getirir."""
    if df_sel.empty:
        st.info("Harita için satır bulunamadı.")
        return

    # Tooltip için güvenli kolon adları
    tip = df_sel.copy()
    if "911_request_count_daily(before_24_hours)" in tip.columns:
        tip = tip.rename(columns={"911_request_count_daily(before_24_hours)": "_911d"})
    else:
        tip["_911d"] = np.nan
    tip["nbr24"] = tip["neighbor_crime_24h"] if "neighbor_crime_24h" in tip.columns else np.nan
    tip["nbr72"] = tip["neighbor_crime_72h"] if "neighbor_crime_72h" in tip.columns else np.nan
    tip["nbr7d"] = tip["neighbor_crime_7d"] if "neighbor_crime_7d" in tip.columns else np.nan

    layers = []

    # Y=1 (kırmızı)
    y1 = tip[tip["Y_label"] == 1] if "Y_label" in tip.columns else tip.iloc[0:0]
    if len(y1):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=y1,
                get_position='[longitude, latitude]',
                get_radius=35,
                pickable=True,
                get_fill_color=[220, 20, 60, 180],
                radius_min_pixels=2,
            )
        )

    # Y=0 (gri) — opsiyonel ve seyrekleştirilmiş
    if show_y0 and "Y_label" in tip.columns:
        y0 = tip[tip["Y_label"] == 0]
        if len(y0) > 50000:
            y0 = y0.sample(50000, random_state=42)
        if len(y0):
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=y0,
                    get_position='[longitude, latitude]',
                    get_radius=25,
                    pickable=False,
                    get_fill_color=[120, 120, 120, 70],
                    radius_min_pixels=1,
                )
            )

    tooltip = {
        "html": """
        <b>GEOID:</b> {GEOID}<br/>
        <b>Y_label:</b> {Y_label}<br/>
        <b>POI risk:</b> {poi_risk_score}<br/>
        <b>911 (24h):</b> {_911d}<br/>
        <b>NBR 24/72/7d:</b> {nbr24}/{nbr72}/{nbr7d}<br/>
        <b>crime_count:</b> {crime_count}
        """,
        "style": {"backgroundColor": "rgba(0,0,0,0.75)", "color": "white"}
    }

    view = pdk.ViewState(latitude=MAP_INITIAL["lat"], longitude=MAP_INITIAL["lon"], zoom=MAP_INITIAL["zoom"])
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)


# =============== UI ===============
st.title(PAGE_TITLE)
st.caption(
    "Bu sayfa **gözlenen Y_label** haritasını gösterir. "
    "İsterseniz **Tahmin (Stacking)** modunu açıp belirli saat için olasılık üretebilirsiniz."
)

df = load_data_any(DATA_PATH)
if df.empty:
    st.error(f"Veri boş ya da yüklenemedi: {DATA_PATH}")
    st.stop()

# Mod seçimi
mode = st.radio("Mod", ["Gözlenen (Y_label)", "Tahmin (Stacking)"], index=0)

# Tarih-saat seçimi
dates = sorted(df["date"].dropna().astype(str).unique())
if not dates:
    st.error("Veride 'date' kolonu yok ya da boş.")
    st.stop()

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    st.write("**Tarih**")
    sel_date = st.selectbox("", options=dates, index=len(dates) - 1, label_visibility="collapsed")
with col_b:
    st.write("**Saat**")
    hours = list(range(24))
    existing_hours = set(df.get("event_hour", pd.Series([], dtype="int")).dropna().astype(int).unique())
    default_hour = 15 if 15 in existing_hours else (max(existing_hours) if existing_hours else 0)
    sel_hour = st.selectbox("", options=hours, index=default_hour, label_visibility="collapsed")
with col_c:
    show_y0 = st.checkbox("Y=0 noktaları da göster", value=False)

# Filtrele
mask = (df["date"].astype(str) == str(sel_date))
if "event_hour" in df.columns:
    mask &= (df["event_hour"].astype(int) == int(sel_hour))
df_sel = df.loc[mask].copy()

st.markdown(f"Seçim: **{sel_date}** · **{sel_hour:02d}:00** — Satır: **{len(df_sel):,}**")
kpi_block(df_sel)

# Harita (gözlenen)
make_map(df_sel, show_y0=show_y0)

# =============== Tahmin (tek GEOID) ===============
if mode == "Tahmin (Stacking)":
    st.markdown("---")
    st.subheader("Tahmin (Stacking) — Tek GEOID")
    if not _PRED_POINT_OK:
        st.warning("Tek nokta tahmin için `serve.py` ve eğitilmiş modeller (models/*.joblib) gerekir.")
    else:
        default_geoid = df_sel["GEOID"].iloc[0] if len(df_sel) else df["GEOID"].iloc[0]
        geoid_sel = st.text_input("GEOID", value=str(default_geoid))
        dt = datetime.strptime(f"{sel_date} {int(sel_hour):02d}:00", "%Y-%m-%d %H:%M")
        with st.spinner("Tahmin üretiliyor..."):
            try:
                res = predict_point(df, geoid_sel, dt)  # df: t-1'e kadar geçmiş olarak kullanılır
                if res.get("top3"):
                    top1_cat, top1_p = res["top3"][0]
                    st.success(f"P(Y=1) = {res['p_occ']:.3f} | Top-1 kategori: {top1_cat} ({top1_p:.3f})")
                else:
                    st.success(f"P(Y=1) = {res['p_occ']:.3f}")
            except Exception as e:
                st.error(f"Tahmin sırasında hata: {e}")

# =============== Tahmin (tüm GEOID’ler) ===============
if mode == "Tahmin (Stacking)":
    st.markdown("---")
    st.subheader("Tahmin (Stacking) — Tüm GEOID'ler")
    if not _PRED_MANY_OK:
        st.info("Toplu tahmin için `serve.py` ve eğitilmiş modeller (models/*.joblib) gerekli.")
    else:
        # Bu saat için tahmin yapılacak GEOID listesi
        if len(df_sel):
            geoids_all = df_sel["GEOID"].astype(str).unique().tolist()
        else:
            geoids_all = df["GEOID"].astype(str).unique().tolist()

        limit = st.slider(
            "Bu saat için tahmin yapılacak maksimum GEOID sayısı",
            min_value=500, max_value=20000, value=min(5000, len(geoids_all))
        )
        geoids = geoids_all[:limit]

        dt = datetime.strptime(f"{sel_date} {int(sel_hour):02d}:00", "%Y-%m-%d %H:%M")
        run_pred = st.button("Seçilen saat için tüm GEOID'lerde tahmin üret")
        if run_pred:
            with st.spinner("Tahminler hesaplanıyor..."):
                pred_df = predict_many(df, geoids, dt)  # df: t-1'e kadar history

                # görselleştirme renkleri (0..1 -> yeşil→kırmızı)
                p = pred_df["p_occ"].fillna(0).clip(0, 1)
                pred_df["r"] = (255 * p).astype(int)
                pred_df["g"] = (255 * (1 - p)).astype(int)
                pred_df["b"] = 40

                # konum birleştir
                pred_vis = pred_df.merge(
                    df[["GEOID", "latitude", "longitude"]].drop_duplicates("GEOID"),
                    on="GEOID", how="left"
                ).dropna(subset=["latitude", "longitude"])

            st.success(f"{len(pred_vis):,} GEOID için tahmin üretildi.")

            # Harita: p_occ renk skalası
            layer_pred = pdk.Layer(
                "ScatterplotLayer",
                data=pred_vis,
                get_position='[longitude, latitude]',
                get_radius=35,
                pickable=True,
                get_fill_color='[r,g,b, 180]',
                radius_min_pixels=2,
            )
            tooltip_pred = {
                "html": """
                <b>GEOID:</b> {GEOID}<br/>
                <b>P(Y=1):</b> {p_occ}<br/>
                <b>Top kategori:</b> {top_cat} ({top_cat_p})
                """,
                "style": {"backgroundColor": "rgba(0,0,0,0.75)", "color": "white"}
            }
            view = pdk.ViewState(latitude=MAP_INITIAL["lat"], longitude=MAP_INITIAL["lon"], zoom=MAP_INITIAL["zoom"])
            deck = pdk.Deck(layers=[layer_pred], initial_view_state=view, map_style="light", tooltip=tooltip_pred)
            st.pydeck_chart(deck, use_container_width=True)

            st.caption("Renk: yeşil ≈ düşük olasılık, kırmızı ≈ yüksek olasılık (doğrusal ölçek).")
            st.dataframe(
                pred_vis[["GEOID", "p_occ", "top_cat", "top_cat_p"]]
                .sort_values("p_occ", ascending=False)
                .head(50),
                use_container_width=True
            )
