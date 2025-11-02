# app.py — SUTAM • Gözlenen Risk Haritası (Y_label)
from __future__ import annotations
import os
from datetime import date as date_cls
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ---- Ayarlar
DATA_PATH = os.getenv("SUTAM_DATA_PATH", "data/sf_crime_observed.csv")
PAGE_TITLE = "SUTAM — Gözlenen Risk Haritası (Y_label)"
MAP_INITIAL = {"lat": 37.7749, "lon": -122.4194, "zoom": 11.2}  # SF merkezi; gerekirse değiştir

st.set_page_config(page_title=PAGE_TITLE, layout="wide")

@st.cache_data(show_spinner=True)
def load_data(path: str) -> pd.DataFrame:
    usecols = [
        "GEOID","latitude","longitude","Y_label","date","event_hour",
        # tooltip için isteğe bağlı:
        "category","poi_risk_score","911_request_count_daily(before_24_hours)",
        "neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d","crime_count"
    ]
    # CSV’de olmayan kolonlar sorun olmasın diye low_memory ile ve mevcut olanları çekiyoruz
    df_head = pd.read_csv(path, nrows=1)
    cols = [c for c in usecols if c in df_head.columns]
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    # tip düzeltmeleri
    if "Y_label" in df.columns:
        df["Y_label"] = pd.to_numeric(df["Y_label"], errors="coerce").fillna(0).astype("int8")
    if "event_hour" in df.columns:
        df["event_hour"] = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype("int16")
    if "latitude" in df.columns and "longitude" in df.columns:
        df = df.dropna(subset=["latitude","longitude"])
    # date'i string bırakıyoruz; filtrelemede string karşılaştıracağız
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str)
    return df

def kpi_block(df: pd.DataFrame, sel: pd.DataFrame):
    total = len(sel)
    y1 = int(sel["Y_label"].sum()) if "Y_label" in sel.columns else 0
    rate = (y1 / total * 100) if total else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam satır", f"{total:,}")
    c2.metric("Y=1 (olay var)", f"{y1:,}")
    c3.metric("Y=1 oranı (%)", f"{rate:.2f}")

def make_map(df_sel: pd.DataFrame, show_y0: bool):
    layers = []

    # Y=1 kırmızı noktalar
    y1 = df_sel[df_sel["Y_label"] == 1] if "Y_label" in df_sel.columns else df_sel.iloc[0:0]
    if len(y1):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=y1,
                get_position='[longitude, latitude]',
                get_radius=35,  # metre
                pickable=True,
                get_fill_color=[220, 20, 60, 180],  # kırmızımsı
                radius_min_pixels=2,
            )
        )

    # İsteğe bağlı Y=0 gri noktalar (seyrekleştirerek)
    if show_y0 and "Y_label" in df_sel.columns:
        y0 = df_sel[df_sel["Y_label"] == 0]
        if len(y0) > 50000:
            y0 = y0.sample(50000, random_state=42)  # büyükse incelt
        if len(y0):
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=y0,
                    get_position='[longitude, latitude]',
                    get_radius=25,
                    pickable=False,
                    get_fill_color=[120, 120, 120, 70],  # gri
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
        "style": {"backgroundColor": "rgba(0,0,0,0.7)", "color": "white"}
    }

    # Tooltip alan adları CSV’de yoksa boş gelsin diye rename
    data_for_tip = df_sel.copy()
    if "911_request_count_daily(before_24_hours)" in data_for_tip.columns:
        data_for_tip = data_for_tip.rename(columns={"911_request_count_daily(before_24_hours)":"_911d"})
    else:
        data_for_tip["_911d"] = np.nan
    for col_old, col_new in [
        ("neighbor_crime_24h","nbr24"),
        ("neighbor_crime_72h","nbr72"),
        ("neighbor_crime_7d","nbr7d"),
    ]:
        if col_old in data_for_tip.columns:
            data_for_tip[col_new] = data_for_tip[col_old]
        else:
            data_for_tip[col_new] = np.nan

    view = pdk.ViewState(latitude=MAP_INITIAL["lat"], longitude=MAP_INITIAL["lon"], zoom=MAP_INITIAL["zoom"])
    deck = pdk.Deck(layers=layers, initial_view_state=view, map_style="light", tooltip=tooltip)
    st.pydeck_chart(deck, use_container_width=True)

# ---- UI
st.title(PAGE_TITLE)
st.caption("Bugün **Y_label** ile gözlenen olayı harita üzerinde gösteriyoruz. "
           "Bu ekran ileride aynı UI ile **tahmin (P(Y=1))** gösterecek.")

df = load_data(DATA_PATH)
if df.empty:
    st.error(f"CSV boş görünüyor: {DATA_PATH}")
    st.stop()

# Tarih ve saat seçenekleri
dates = sorted(df["date"].dropna().astype(str).unique())
min_d, max_d = dates[0], dates[-1]
col_a, col_b, col_c = st.columns([2,1,1])
with col_a:
    st.write("**Tarih**")
    sel_date = st.selectbox("", options=dates, index=len(dates)-1, label_visibility="collapsed")
with col_b:
    st.write("**Saat**")
    hours = list(range(24))
    default_hour = hours[-1] if 23 in df.get("event_hour", pd.Series([])).unique() else 0
    sel_hour = st.selectbox("", options=hours, index=default_hour, label_visibility="collapsed")
with col_c:
    show_y0 = st.checkbox("Y=0 noktaları da göster", value=False)

# Filtre
mask = (df["date"].astype(str) == str(sel_date))
if "event_hour" in df.columns:
    mask &= (df["event_hour"].astype(int) == int(sel_hour))
df_sel = df.loc[mask].copy()

st.markdown(f"Seçim: **{sel_date}** · **{sel_hour:02d}:00**  "
            f"(aralık: {min_d} → {max_d}, satır: {len(df_sel):,})")

kpi_block(df, df_sel)

if len(df_sel) == 0:
    st.info("Bu tarih-saat için satır bulunamadı. Başka bir saat seçiniz.")
else:
    make_map(df_sel, show_y0=show_y0)

    # Örnek tablo: Y=1 satırlardan ilk 50 (harita tooltips ile eşleşsin)
    if "Y_label" in df_sel.columns:
        y1_table = df_sel[df_sel["Y_label"] == 1].copy()
        keep_cols = [c for c in [
            "GEOID","latitude","longitude","Y_label","category",
            "poi_risk_score","neighbor_crime_24h","neighbor_crime_72h","neighbor_crime_7d",
            "crime_count"
        ] if c in y1_table.columns]
        st.subheader("Y=1 örnekleri (ilk 50)")
        st.dataframe(y1_table[keep_cols].head(50), use_container_width=True)

