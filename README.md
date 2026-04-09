# Air Quality Monitoring & Forecasting

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-R²%3D0.9987-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time air quality monitoring and ML forecasting system built with Python, pandas, XGBoost, and live API data from 5 global cities.

---

## Results

| Metric | Value | Target |
|--------|-------|--------|
| R² Score | **0.9987** | ≥ 0.85 |
| RMSE | **2.26** | Minimized |
| MAPE | **2.79%** | < 5% |
| Cities | 5 | — |
| Features | 34 | — |

---

## Dashboard

![Dashboard](outputs/dashboard.png)

---

## Architecture
AQICN API ──┐
├──► ingestion.py ──► pipeline.py ──► features.py ──► train.py ──► evaluate.py
OpenWeather ─┘

---

## Project Structure
air-quality-forecast/
├── src/
│   ├── ingestion.py
│   ├── pipeline.py
│   ├── features.py
│   ├── train.py
│   └── evaluate.py
├── data/
├── models/
├── outputs/
└── requirements.txt

---

## Setup

```bash
git clone https://github.com/bsweehoney/air-quality-forecast.git
cd air-quality-forecast
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
AQICN_API_KEY=your_key_here
OPENWEATHER_API_KEY=your_key_here

---

## Run

```bash
python src/ingestion.py
python src/pipeline.py
python src/features.py
python src/train.py
python src/evaluate.py
```

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML Model | XGBoost, scikit-learn |
| Pipeline | pandas, pyarrow |
| Visualization | matplotlib |
| APIs | AQICN, OpenWeatherMap |

---

## Cities Monitored

| City | Avg AQI | Category |
|------|---------|----------|
| Beijing | ~155 | Unhealthy |
| Delhi | ~160 | Unhealthy |
| London | ~38 | Good |
| New York | ~45 | Good |
| Sydney | ~18 | Good |
Then run in terminal:
powershellgit add README.md
git commit -m "Update README"
git push origin main
