import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="Asamannoor Environmental & Health Risk Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Asamannoor Environmental & Health Impact Dashboard (2023–2035)")
st.markdown("Quantifying the long-term ecological and public health risks of the plywood industrial cluster in Asamannoor, Kerala.")
st.divider()

# Load Data and Generate Projections
@st.cache_data
def load_projection_data():
    try:
        df = pd.read_csv("asamannoor_ml_ready.csv", index_col="time", parse_dates=True)
        hist_yearly = df.groupby(df.index.year).mean()
        last_hist_year = hist_yearly.index[-1]
        last_hist_pm25 = hist_yearly.loc[last_hist_year, 'pm2_5']
        last_hist_aqi = hist_yearly.loc[last_hist_year, 'us_aqi']
    except Exception:
        hist_yearly = pd.DataFrame()
        last_hist_year = 2025
        last_hist_pm25 = 28.5
        last_hist_aqi = 85.0

    all_years = list(range(2023, 2036))
    annual_growth_rate = 0.035  # 3.5% unmitigated industrial expansion factor
    projection_data = []

    for year in all_years:
        if not hist_yearly.empty and year in hist_yearly.index:
            pm25_val = hist_yearly.loc[year, 'pm2_5']
            aqi_val = hist_yearly.loc[year, 'us_aqi']
        else:
            years_diff = year - last_hist_year
            pm25_val = last_hist_pm25 * ((1 + annual_growth_rate) ** max(0, years_diff))
            aqi_val = last_hist_aqi * ((1 + (annual_growth_rate * 0.8)) ** max(0, years_diff))

        excess_pm = max(0.0, pm25_val - 5.0)
        year_index_offset = year - 2023

        lung_cancer = min(98.0, 5.0 + (excess_pm / 7.5) * 8.0 + (year_index_offset * 1.5))
        respiratory = min(99.0, 10.0 + (excess_pm / 5.0) * 12.0 + (year_index_offset * 1.2))
        skin = min(95.0, 8.0 + (pm25_val / 15.0) * 20.0 + (year_index_offset * 1.0))
        water_deg = min(100.0, 20.0 + (pm25_val * 0.85) + (year_index_offset * 1.4))
        soil_stress = min(100.0, 12.0 + (pm25_val * 1.05) + (year_index_offset * 1.6))

        projection_data.append({
            'Year': year,
            'Projected AQI': round(aqi_val, 1),
            'PM2.5 (µg/m³)': round(pm25_val, 1),
            'Lung Cancer Risk (%)': round(lung_cancer, 1),
            'Respiratory Disease Risk (%)': round(respiratory, 1),
            'Skin Irritation Risk (%)': round(skin, 1),
            'Water Degradation Index (%)': round(water_deg, 1),
            'Soil Stress Index (%)': round(soil_stress, 1)
        })

    return pd.DataFrame(projection_data)

proj_df = load_projection_data()

# Sidebar Explorer
st.sidebar.header("🔍 Interactive Explorer")
selected_year = st.sidebar.slider("Select Year for Detailed Snapshot", 2023, 2035, 2026)

year_row = proj_df[proj_df['Year'] == selected_year].iloc[0]

st.subheader(f"📊 Impact Snapshot for {selected_year}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Projected AQI", f"{year_row['Projected AQI']}")
col2.metric("PM2.5 Concentration", f"{year_row['PM2.5 (µg/m³)']} µg/m³")
col3.metric("Lung Cancer Risk", f"{year_row['Lung Cancer Risk (%)']}%")
col4.metric("Respiratory Risk", f"{year_row['Respiratory Disease Risk (%)']}%")

col5, col6 = st.columns(2)
col5.metric("Water Degradation Index", f"{year_row['Water Degradation Index (%)']}%")
col6.metric("Soil Stress Index", f"{year_row['Soil Stress Index (%)']}%")

st.divider()

# Interactive Chart Section
st.subheader("📈 13-Year Longitudinal Trend Analysis (2023–2035)")
metric_choice = st.selectbox(
    "Select metric to inspect decade trajectory:",
    ['Projected AQI', 'Lung Cancer Risk (%)', 'Respiratory Disease Risk (%)', 'Skin Irritation Risk (%)', 'Water Degradation Index (%)', 'Soil Stress Index (%)']
)

color_dict = {
    'Projected AQI': '#337ab7',
    'Lung Cancer Risk (%)': '#b52a0a',
    'Respiratory Disease Risk (%)': '#d9534f',
    'Skin Irritation Risk (%)': '#f0ad4e',
    'Water Degradation Index (%)': '#5bc0de',
    'Soil Stress Index (%)': '#5cb85c'
}

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(proj_df['Year'].astype(str), proj_df[metric_choice], color=color_dict[metric_choice], edgecolor='black', alpha=0.85, width=0.6)

ax.set_title(f'Asamannoor Timeline & Forecast: {metric_choice}', fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel('Year', fontsize=10)
ax.set_ylabel(metric_choice, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.6)

suffix = '%' if '%' in metric_choice else ''
y_max = ax.get_ylim()[1]
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + (y_max * 0.02),
        f'{height}{suffix}',
        ha='center',
        va='bottom',
        fontsize=8,
        fontweight='bold'
    )

st.pyplot(fig)
