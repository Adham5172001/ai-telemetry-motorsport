"""
Telemetry Feature Engineering
Author: Adham Aboulkheir | Saudi Motorsport Company
"""
import numpy as np
import pandas as pd
from typing import List, Dict


TELEMETRY_CHANNELS = [
    "throttle", "brake", "speed_kmh", "rpm", "lat_g", "long_g",
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "engine_temp", "brake_temp_fl", "brake_temp_fr"
]


def compute_rolling_features(df: pd.DataFrame,
                               channels: List[str] = None,
                               windows: List[int] = [10, 50, 200]) -> pd.DataFrame:
    """Compute rolling statistics for telemetry channels."""
    if channels is None:
        channels = TELEMETRY_CHANNELS
    
    result = df.copy()
    for ch in channels:
        if ch not in df.columns:
            continue
        for w in windows:
            result[f"{ch}_mean_{w}"] = df[ch].rolling(w, min_periods=1).mean()
            result[f"{ch}_std_{w}"]  = df[ch].rolling(w, min_periods=1).std().fillna(0)
            result[f"{ch}_max_{w}"]  = df[ch].rolling(w, min_periods=1).max()
    
    return result


def extract_lap_features(lap_df: pd.DataFrame) -> Dict[str, float]:
    """Extract aggregate features from a single lap's telemetry."""
    features = {}
    
    for ch in TELEMETRY_CHANNELS:
        if ch not in lap_df.columns:
            continue
        features[f"{ch}_mean"] = lap_df[ch].mean()
        features[f"{ch}_max"]  = lap_df[ch].max()
        features[f"{ch}_std"]  = lap_df[ch].std()
    
    # Derived features
    features["mean_tyre_temp"] = lap_df[["tyre_temp_fl", "tyre_temp_fr",
                                          "tyre_temp_rl", "tyre_temp_rr"]].mean().mean()
    features["tyre_temp_balance"] = (lap_df["tyre_temp_fl"].mean() -
                                      lap_df["tyre_temp_rr"].mean())
    features["throttle_brake_overlap"] = ((lap_df["throttle"] > 10) &
                                           (lap_df["brake"] > 10)).mean()
    features["high_speed_pct"] = (lap_df["speed_kmh"] > 200).mean()
    features["braking_events"] = (lap_df["brake"].diff() > 20).sum()
    
    return features


def extract_all_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features for each lap."""
    lap_features = []
    for lap_num in df["lap"].unique():
        lap_df = df[df["lap"] == lap_num]
        features = extract_lap_features(lap_df)
        features["lap"] = lap_num
        lap_features.append(features)
    return pd.DataFrame(lap_features)


if __name__ == "__main__":
    from pipeline.ingest import generate_race_telemetry
    
    print("Feature Engineering Demo")
    df = generate_race_telemetry(n_laps=10, samples_per_lap=500)
    lap_features = extract_all_lap_features(df)
    print(f"Lap features shape: {lap_features.shape}")
    print(f"Features: {[c for c in lap_features.columns if c != 'lap'][:8]}")
