import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your feature-engineered dataset containing historical data from 2023
df = pd.read_csv("asamannoor_ml_ready.csv", index_col="time", parse_dates=True)

# Extract actual historical yearly averages from the dataset
hist_yearly = df.groupby(df.index.year).mean()

# 2. Define the complete timeline from 2023 to 2035
all_years = list(range(2023, 2036))
annual_growth_rate = 0.035  # 3.5% yearly increase for unmitigated future expansion

projection_data = []

# Baseline reference from historical data
last_hist_year = hist_yearly.index[-1]
last_hist_pm25 = hist_yearly.loc[last_hist_year, 'pm2_5']
last_hist_aqi = hist_yearly.loc[last_hist_year, 'us_aqi']

for i, year in enumerate(all_years):
    if year in hist_yearly.index:
        # Use actual recorded historical data
        pm25_val = hist_yearly.loc[year, 'pm2_5']
        aqi_val = hist_yearly.loc[year, 'us_aqi']
    else:
        # Project forward for future years past recorded data
        years_diff = year - last_hist_year
        pm25_val = last_hist_pm25 * ((1 + annual_growth_rate) ** years_diff)
        aqi_val = last_hist_aqi * ((1 + (annual_growth_rate * 0.8)) ** years_diff)

    excess_pm = max(0.0, pm25_val - 5.0)
    year_index_offset = year - 2023

    # Calculate multi-factor health and environmental risk percentages
    lung_cancer = min(98.0, 5.0 + (excess_pm / 7.5) * 8.0 + (year_index_offset * 1.5))
    respiratory = min(99.0, 10.0 + (excess_pm / 5.0) * 12.0 + (year_index_offset * 1.2))
    skin = min(95.0, 8.0 + (pm25_val / 15.0) * 20.0 + (year_index_offset * 1.0))
    water_deg = min(100.0, 20.0 + (pm25_val * 0.85) + (year_index_offset * 1.4))
    soil_stress = min(100.0, 12.0 + (pm25_val * 1.05) + (year_index_offset * 1.6))

    projection_data.append({
        'Year': str(year),
        'Projected AQI': round(aqi_val, 1),
        'Lung Cancer Risk (%)': round(lung_cancer, 1),
        'Respiratory Disease Risk (%)': round(respiratory, 1),
        'Skin Irritation Risk (%)': round(skin, 1),
        'Water Degradation Index (%)': round(water_deg, 1),
        'Soil Stress Index (%)': round(soil_stress, 1)
    })

proj_df = pd.DataFrame(projection_data)

# 3. Generate separate figure windows and save individual files for each metric (2023–2035)
metrics = [
    ('Projected AQI', 'US AQI Score', '#337ab7', 200, 'aqi_2023_2035'),
    ('Lung Cancer Risk (%)', 'Risk Probability (%)', '#b52a0a', 110, 'lung_cancer_2023_2035'),
    ('Respiratory Disease Risk (%)', 'Risk Probability (%)', '#d9534f', 110, 'respiratory_2023_2035'),
    ('Skin Irritation Risk (%)', 'Risk Probability (%)', '#f0ad4e', 110, 'skin_2023_2035'),
    ('Water Degradation Index (%)', 'Degradation Score (%)', '#5bc0de', 110, 'water_2023_2035'),
    ('Soil Stress Index (%)', 'Stress Score (%)', '#5cb85c', 110, 'soil_2023_2035')
]

for col_name, y_label, color, y_max, filename_tag in metrics:
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(proj_df['Year'], proj_df[col_name], color=color, edgecolor='black', alpha=0.85, width=0.6)
    
    ax.set_title(f'Asamannoor Timeline & Forecast: {col_name} (2023–2035)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Year', fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_ylim(0, y_max)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    suffix = '%' if '%' in col_name else ''
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
        
    plt.tight_layout()
    plt.savefig(f'asamannoor_{filename_tag}_bar_graph.png', dpi=300)

print("All 2023–2035 bar graphs successfully generated in separate windows and saved to your project folder!")
plt.show()
