import requests, os, json, time, random
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

AQICN_KEY = os.getenv("AQICN_API_KEY")
CITIES = ["new york", "london", "beijing", "delhi", "sydney"]

def fetch_aqi(city: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={AQICN_KEY}"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json()
    if data["status"] != "ok":
        return {}
    d = data["data"]
    iaqi = d.get("iaqi", {})
    return {
        "city":      city,
        "timestamp": datetime.utcnow().isoformat(),
        "aqi":       d.get("aqi"),
        "pm25":      iaqi.get("pm25", {}).get("v"),
        "pm10":      iaqi.get("pm10", {}).get("v"),
        "no2":       iaqi.get("no2",  {}).get("v"),
        "o3":        iaqi.get("o3",   {}).get("v"),
        "co":        iaqi.get("co",   {}).get("v"),
        "so2":       iaqi.get("so2",  {}).get("v"),
    }

def fake_weather(city: str) -> dict:
    """Synthetic weather until OpenWeatherMap key activates."""
    profiles = {
        "new york":  {"temperature": 12, "humidity": 60, "wind_speed": 4.5, "pressure": 1013},
        "london":    {"temperature":  8, "humidity": 75, "wind_speed": 5.2, "pressure": 1008},
        "beijing":   {"temperature":  5, "humidity": 50, "wind_speed": 3.1, "pressure": 1020},
        "delhi":     {"temperature": 28, "humidity": 40, "wind_speed": 2.8, "pressure": 1005},
        "sydney":    {"temperature": 22, "humidity": 65, "wind_speed": 6.0, "pressure": 1015},
    }
    base = profiles.get(city, {"temperature": 15, "humidity": 60, "wind_speed": 4.0, "pressure": 1013})
    return {
        "temperature": base["temperature"] + random.uniform(-2, 2),
        "humidity":    base["humidity"]    + random.uniform(-5, 5),
        "wind_speed":  base["wind_speed"]  + random.uniform(-1, 1),
        "pressure":    base["pressure"]    + random.uniform(-3, 3),
        "weather":     random.choice(["Clear", "Clouds", "Rain"]),
    }

def collect_all(cities=CITIES) -> list:
    records = []
    for city in cities:
        try:
            aqi     = fetch_aqi(city)
            if not aqi:
                print(f"✗ {city}: no AQI data")
                continue
            weather = fake_weather(city)
            record  = {**aqi, **weather}
            records.append(record)
            print(f"✓ {city}: AQI={record.get('aqi')} | Temp={record.get('temperature'):.1f}°C")
        except Exception as e:
            print(f"✗ {city}: {e}")
        time.sleep(0.5)
    return records

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    data = collect_all()
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M")
    path = f"data/raw/snapshot_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nSaved {len(data)} records → {path}")