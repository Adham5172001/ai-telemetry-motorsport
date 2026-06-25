# AI Telemetry Analysis — Motorsport Performance

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A real-time machine learning system for analysing motorsport telemetry data to optimise vehicle performance, predict component failures, and provide actionable insights to race engineers during live events.

## Overview

Modern motorsport vehicles generate terabytes of telemetry data per race — throttle position, brake pressure, tyre temperatures, suspension travel, GPS coordinates, and hundreds of other channels sampled at up to 1,000 Hz. This system processes that data in real time to surface performance insights that would take human analysts hours to identify.

## Features

- **Real-time anomaly detection**: Flags unusual sensor readings that may indicate component failure
- **Lap time prediction**: Predicts optimal lap time based on current conditions
- **Tyre degradation modelling**: Forecasts tyre performance over remaining laps
- **Sector-by-sector comparison**: Compares driver performance against optimal reference laps
- **Predictive maintenance**: Identifies components at risk of failure before they fail

## Models

| Task | Model | Performance |
|------|-------|-------------|
| Anomaly detection | Isolation Forest | Precision: 0.91, Recall: 0.87 |
| Lap time prediction | Gradient Boosting | MAE: 0.34s |
| Tyre degradation | LSTM (48 units) | RMSE: 0.021 (normalised) |
| Component failure | Random Forest | AUC-ROC: 0.94 |

## Architecture

```
Telemetry Stream (CAN bus / UDP)
        │
  Real-time Ingestion (Apache Kafka)
        │
  Feature Engineering Pipeline
  ├── Rolling statistics (1s, 5s, 30s windows)
  ├── Sector segmentation
  └── Reference lap comparison
        │
  ML Inference Engine (parallel models)
        │
  Race Engineer Dashboard (WebSocket)
```

## Installation

```bash
git clone https://github.com/Adham5172001/ai-telemetry-motorsport.git
cd ai-telemetry-motorsport

# Using Docker (recommended)
docker-compose up -d

# Or manual installation
pip install -r requirements.txt
python pipeline/ingest.py --source udp --port 20777
```

## Project Structure

```
ai-telemetry-motorsport/
├── pipeline/
│   ├── ingest.py           # Data ingestion
│   ├── features.py         # Feature engineering
│   └── inference.py        # Real-time ML inference
├── models/
│   ├── anomaly.py          # Isolation Forest
│   ├── lap_prediction.py   # Gradient Boosting
│   ├── tyre_model.py       # LSTM degradation model
│   └── maintenance.py      # Failure prediction
├── dashboard/
│   └── app.py              # WebSocket dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## License

MIT License
