# app.py — SUTAM • Gözlenen Risk Haritası (Y_label) + Tahmin (Stacking) Modu
from __future__ import annotations

import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

# (opsiyonel) serve.py'den tahmin fonksiyonunu içe aktar
try:
    from serve import predict_point  # models/ klasörüyle birlikte çalışır
    _PRED_AVAILABLE = True
except Exception:
    _PRED_AVAILABLE = False

# ---- Ayarlar
DATA_PATH = os.getenv("SUTAM_DATA_PATH", "data/sf_crime_observed.csv")
PAGE_TITLE = "SUTAM — Gözlenen Risk Haritası (Y_label)"
MAP_INITIAL = {"lat": 37.7749, "lon": -122.4194, "zoom": 11.2}  # SF merkezi

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    """CSV'yi yükler; mevcut olan kullanışlı kolonları alır ve tipleri düzenler."""
    fallback_cols = [
        "GEOID","latitude","longitude","Y_label","date","event_hour",
        "category","poi_risk_score","911_request_count_daily(before_24_hours)",
        "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d","crime_count"
    ]
    try:
        head = pd.read_csv(path, nrows=1)
        usecols = [c for c in fallback_cols if c in head.columns]
    except Exception as e:
        st.error(f"CSV okunamadı: {path}\n{e}")
        return pd.DataFrame()

    df = pd.read_csv(path, usecols=usecols, low_memory=False)

    if "Y_label" in df.columns:
        df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    if "event_hour" in df.columns:
        df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude","longitude"])
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df

def kpi_block(df_all: pd.DataFrame, df_sel: pd.DataFrame):
    total = len(df_sel)
    y1 = int(df_sel["Y_label"].sum()) if "Y_label" in df_sel.columns and total else 0
    rate = (100.0 * y1 / total) if total else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Satır (seçim)", f"{total:,}")
    c2.metric("Y=1 (olay)", f"{y1:,}")
    c3.metric("Y=1 oranı (%)", f"{rate:.2f}")

def make_map(df_sel: pd.DataFrame, show_y0: bool):
    layers = []

    # Y=1 kırmızı
    if "Y_label" in df_sel.columns:
        y1 = df_sel[df_sel["Y_label"] == 1]
    else:
        y1 = df_sel.iloc[0:0]
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

    # Y=0 gri (opsiyonel, seyrekleştir)
    if show_y0 and "Y_label" in df_sel.columns:
        y0 = df_sel[df_sel["Y_label"] == 0]
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

    # Tooltip alanlarını güvenli isimlere taşı
    data_tip = df_sel.copy()
    if "911_request_count_daily(before_24_hours)" in data_tip.columns:
        data_tip = data_tip.rename(columns={"911_request_count_daily(before_24_hours)": "_911d"})
    else:
        data_tip["_911d"] = np.nan
    for src, dst in [("neighbor_crime_24h","nbr24"),
                     ("neighbor_crime_72h","nbr72"),
                     ("neighbor_crime_7d","nbr7d")]:
        data_tip[dst] = data_tip[src] if src in data_tip.columns else np.nan

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
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip=tooltip, data=data_tip)
    st.pydeck_chart(deck, use_container_width=True)

# -------- UI --------
st.title(PAGE_TITLE)
st.caption("Bugün **Y_label** (gözlenen) haritasını gösteriyoruz. "
           "İsterseniz **Tahmin (Stacking)** modunu açıp tek GEOID için olasılık alabilirsiniz.")

df = load_data(DATA_PATH)
if df.empty:
    st.error(f"CSV boş ya da yüklenemedi: {DATA_PATH}")
    st.stop()

# Mod seçimi
mode = st.radio("Mod", ["Gözlenen (Y_label)", "Tahmin (Stacking)"], index=0)

# Tarih-saat seçimi
dates = sorted(df["date"].dropna().astype(str).unique())
if not dates:
    st.error("Veride 'date' kolonu yok ya da boş.")
    st.stop()

col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    st.write("**Tarih**")
    sel_date = st.selectbox("", options=dates, index=len(dates)-1, label_visibility="collapsed")
with col_b:
    st.write("**Saat**")
    hours = list(range(24))
    # varsa df'teki saatlerden birini varsayılan seç
    existing_hours = set(df.get("event_hour", pd.Series([], dtype="int")).dropna().astype(int).unique())
    default_hour = 15 if 15 in existing_hours else (max(existing_hours) if existing_hours else 0)
    sel_hour = st.selectbox("", options=hours, index=default_hour, label_visibility="collapsed")
with col_c:
    show_y0 = st.checkbox("Y=0 noktaları da göster", value=False)

# Filtre
mask = (df["date"].astype(str) == str(sel_date))
if "event_hour" in df.columns:
    mask &= (df["event_hour"].astype(int) == int(sel_hour))
df_sel = df.loc[mask].copy()

st.markdown(f"Seçim: **{sel_date}** · **{sel_hour:02d}:00**  — Satır: **{len(df_sel):,}**")
kpi_block(df, df_sel)

# Harita (gözlenen)
make_map(df_sel, show_y0=show_y0)

# Alt tablo (örnekler)
if len(df_sel) and "Y_label" in df_sel.columns:
    y1 = df_sel[df_sel["Y_label"] == 1]
    keep_cols = [c for c in [
        "GEOID","latitude","longitude","Y_label","category",
        "poi_risk_score","neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
        "crime_count"
    ] if c in df_sel.columns]
    st.subheader("Y=1 örnekleri (ilk 50)")
    st.dataframe(y1[keep_cols].head(50), use_container_width=True)

# Tahmin modu
if mode == "Tahmin (Stacking)":
    st.subheader("Tahmin (Stacking)")
    if not _PRED_AVAILABLE:
        st.warning("Tahmin için `serve.py` ve eğitilmiş modeller (models/*.joblib) gerekiyor. "
                   "Önce eğitim adımlarını çalıştırın.")
    else:
        # Kullanıcıdan tek GEOID al
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
