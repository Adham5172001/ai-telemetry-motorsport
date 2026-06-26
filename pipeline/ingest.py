"""
Telemetry Data Ingestion
Author: Adham Aboulkheir | Saudi Motorsport Company
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class TelemetryFrame:
    """Single telemetry data frame from one time step."""
    timestamp: float
    lap: int
    throttle: float       # 0-100%
    brake: float          # 0-100%
    speed_kmh: float
    rpm: int
    gear: int
    steering_angle: float # degrees
    lat_g: float
    long_g: float
    tyre_temp_fl: float
    tyre_temp_fr: float
    tyre_temp_rl: float
    tyre_temp_rr: float
    fuel_remaining: float # kg
    engine_temp: float    # °C
    brake_temp_fl: float
    brake_temp_fr: float


def generate_race_telemetry(n_laps: int = 50,
                             samples_per_lap: int = 1000,
                             seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic race telemetry data.
    
    Simulates a complete race with:
    - Tyre degradation over laps
    - Fuel burn
    - Varying driving styles per sector
    - Random anomalies (5% of laps)
    """
    np.random.seed(seed)
    frames = []
    
    for lap in range(n_laps):
        t = np.linspace(0, 90, samples_per_lap)
        tyre_wear = lap / n_laps  # 0 to 1
        fuel_load = 1.0 - (lap / n_laps) * 0.85
        
        # Simulate sectors: straight, corner, chicane
        throttle_base = 70 + 20 * np.sin(t / 5) + 10 * np.cos(t / 3)
        brake_base    = 15 + 15 * np.cos(t / 4) + 5 * np.sin(t / 2)
        speed_base    = 180 + 60 * np.sin(t / 8) + 20 * np.cos(t / 4)
        rpm_base      = 8000 + 2000 * np.sin(t / 6) + 500 * np.cos(t / 3)
        
        # Add tyre degradation effect
        grip_factor = 1.0 - 0.2 * tyre_wear
        speed_base *= grip_factor
        
        # Add anomaly for 5% of laps
        is_anomaly = np.random.random() < 0.05
        if is_anomaly:
            anomaly_start = np.random.randint(200, 700)
            throttle_base[anomaly_start:anomaly_start+50] *= 0.3
            rpm_base[anomaly_start:anomaly_start+50] *= 0.7
        
        lap_frames = pd.DataFrame({
            "timestamp":    t + lap * 90,
            "lap":          lap + 1,
            "throttle":     np.clip(throttle_base + np.random.normal(0, 3, samples_per_lap), 0, 100),
            "brake":        np.clip(brake_base + np.random.normal(0, 2, samples_per_lap), 0, 100),
            "speed_kmh":    np.clip(speed_base + np.random.normal(0, 5, samples_per_lap), 30, 340),
            "rpm":          np.clip(rpm_base + np.random.normal(0, 150, samples_per_lap), 2000, 14000).astype(int),
            "gear":         np.clip((speed_base / 50).astype(int), 1, 8),
            "steering_angle": 15 * np.sin(t / 3) + np.random.normal(0, 2, samples_per_lap),
            "lat_g":        1.8 * np.sin(t / 3) * grip_factor + np.random.normal(0, 0.15, samples_per_lap),
            "long_g":       0.9 * np.cos(t / 4) + np.random.normal(0, 0.12, samples_per_lap),
            "tyre_temp_fl": 85 + 15 * tyre_wear + 10 * np.sin(t / 20) + np.random.normal(0, 2, samples_per_lap),
            "tyre_temp_fr": 87 + 15 * tyre_wear + 10 * np.sin(t / 20) + np.random.normal(0, 2, samples_per_lap),
            "tyre_temp_rl": 90 + 18 * tyre_wear + 12 * np.sin(t / 18) + np.random.normal(0, 2, samples_per_lap),
            "tyre_temp_rr": 88 + 16 * tyre_wear + 11 * np.sin(t / 19) + np.random.normal(0, 2, samples_per_lap),
            "fuel_remaining": fuel_load * 110 - np.linspace(0, 2.2, samples_per_lap),
            "engine_temp":  95 + 10 * np.sin(t / 30) + np.random.normal(0, 1.5, samples_per_lap),
            "brake_temp_fl": 200 + 150 * np.abs(np.sin(t / 4)) + np.random.normal(0, 10, samples_per_lap),
            "brake_temp_fr": 195 + 145 * np.abs(np.sin(t / 4)) + np.random.normal(0, 10, samples_per_lap),
            "is_anomaly":   is_anomaly,
        })
        frames.append(lap_frames)
    
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    print("Telemetry Ingestion Demo")
    df = generate_race_telemetry(n_laps=10, samples_per_lap=500)
    print(f"Generated: {len(df)} telemetry frames across {df['lap'].nunique()} laps")
    print(f"Channels: {list(df.columns)}")
    print(f"\nSample statistics:")
    print(df[["speed_kmh", "throttle", "brake", "tyre_temp_fl"]].describe().round(1).to_string())
