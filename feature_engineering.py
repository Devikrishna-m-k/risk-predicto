import pandas as pd

# 1. Load your downloaded hourly CSV file
df = pd.read_csv("asamannoor_air_quality.csv")

# 2. Convert 'time' column to proper datetime format and sort chronologically
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').set_index('time')

# 3. Resample hourly data into DAILY averages ('D')
# This turns 24 hourly rows per day into 1 clean daily summary row
daily_df = df.resample('D').mean()

# 4. Handle any missing values (forward-fill followed by backward-fill)
daily_df = daily_df.ffill().bfill()

# 5. Extract calendar/seasonal clues (helps the model learn monsoon vs dry months)
daily_df['month'] = daily_df.index.month
daily_df['day_of_week'] = daily_df.index.dayofweek
daily_df['day_of_year'] = daily_df.index.dayofyear

# 6. Create Lag Features (Yesterday's and last week's pollution levels)
# Using .shift(1) brings yesterday's values onto today's row without leaking today's data
daily_df['pm2_5_lag_1d'] = daily_df['pm2_5'].shift(1)
daily_df['pm10_lag_1d'] = daily_df['pm10'].shift(1)
daily_df['aqi_lag_1d'] = daily_df['us_aqi'].shift(1)
daily_df['aqi_lag_7d'] = daily_df['us_aqi'].shift(7)

# 7. Create Rolling Averages (7-day and 30-day moving averages)
# Note: .shift(1) is applied before .rolling() so today's actual value isn't included in its own prediction window
daily_df['pm2_5_roll_7d'] = daily_df['pm2_5'].shift(1).rolling(window=7).mean()
daily_df['pm2_5_roll_30d'] = daily_df['pm2_5'].shift(1).rolling(window=30).mean()

daily_df['aqi_roll_7d'] = daily_df['us_aqi'].shift(1).rolling(window=7).mean()
daily_df['aqi_roll_30d'] = daily_df['us_aqi'].shift(1).rolling(window=30).mean()

# 8. Drop the first 30 rows (which will contain NaN because a 30-day average needs 30 prior days)
daily_df = daily_df.dropna()

# 9. Save the ready-to-train dataset
daily_df.to_csv("asamannoor_ml_ready.csv")
print("Feature engineering complete! Saved to 'asamannoor_ml_ready.csv'")
print(f"Total clean daily rows ready for ML: {len(daily_df)}")
print(daily_df[['pm2_5', 'pm2_5_lag_1d', 'pm2_5_roll_7d', 'us_aqi']].head())
