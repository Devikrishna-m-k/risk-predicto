import pandas as pd
import matplotlib.pyplot as plt

# 1. Load your ML-ready dataset
df = pd.read_csv("asamannoor_ml_ready.csv", index_col="time", parse_dates=True)

# 2. Set up the visualization style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, ax = plt.subplots(figsize=(12, 6))

# 3. Plot PM2.5 levels over time
ax.plot(df.index, df['pm2_5'], label='Daily PM2.5 (µg/m³)', color='#d9534f', linewidth=1.5, alpha=0.8)

# 4. Add WHO Safe Limit Reference Line (15 µg/m³ threshold)
ax.axhline(y=15, color='#f0ad4e', linestyle='--', linewidth=2, label='WHO 24-Hour Safe Limit (15 µg/m³)')

# 5. Formatting the chart
ax.set_title('Asamannoor Particulate Matter (PM2.5) Historical Trend Analysis', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Timeline', fontsize=11)
ax.set_ylabel('Concentration (µg/m³)', fontsize=11)
ax.legend(loc='upper left', frameon=True)

# 6. Save high-resolution image for your Millennium Fellowship report
plt.tight_layout()
plt.savefig('asamannoor_pm25_trend.png', dpi=300)
print("Chart successfully generated and saved as 'asamannoor_pm25_trend.png' in your project folder!")

# Display the chart window
plt.show()
