import requests
import pandas as pd

# Coordinates for Asamannoor, Ernakulam
LAT = 10.0754
LON = 76.5056

# 1. Use the dedicated Air Quality API endpoint
url = (
    f"https://air-quality-api.open-meteo.com/v1/air-quality?"
    f"latitude={LAT}&longitude={LON}"
    f"&hourly=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
    f"&start_date=2023-01-01&end_date=2026-11-15"
)

# 2. Fetch data from the API
response = requests.get(url)
data = response.json()

# 3. Check if 'hourly' data exists in the response
if "hourly" in data:
    df = pd.DataFrame(data["hourly"])
    
    # 4. Save to CSV
    df.to_csv("asamannoor_air_quality.csv", index=False)
    print("CSV file successfully downloaded and saved!")
    print(df.head())
else:
    print("Error fetching data:", data)
