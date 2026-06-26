"""
Telemetry Anomaly Detection
Author: Adham Aboulkheir | Saudi Motorsport Company
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List


class TelemetryAnomalyDetector:
    """
    Isolation Forest-based anomaly detection for telemetry channels.
    Detects unusual sensor readings that may indicate component failure.
    """
    
    FEATURE_CHANNELS = [
        "throttle", "brake", "speed_kmh", "rpm",
        "lat_g", "long_g", "tyre_temp_fl", "tyre_temp_rr",
        "engine_temp", "brake_temp_fl"
    ]
    
    def __init__(self, contamination: float = 0.03, random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=random_state
        )
        self.scaler = StandardScaler()
        self._fitted = False
    
    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.FEATURE_CHANNELS if c in df.columns]
        return df[cols].fillna(df[cols].mean()).values
    
    def fit(self, df: pd.DataFrame) -> "TelemetryAnomalyDetector":
        X = self._get_features(df)
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s)
        self._fitted = True
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Returns -1 for anomalies, +1 for normal."""
        X = self._get_features(df)
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)
    
    def anomaly_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Returns anomaly scores (lower = more anomalous)."""
        X = self._get_features(df)
        X_s = self.scaler.transform(X)
        return self.model.score_samples(X_s)
    
    def get_anomaly_report(self, df: pd.DataFrame) -> Dict:
        """Generate anomaly report with channel-level analysis."""
        predictions = self.predict(df)
        scores = self.anomaly_scores(df)
        anomaly_mask = predictions == -1
        
        report = {
            "total_frames": len(df),
            "anomaly_count": int(anomaly_mask.sum()),
            "anomaly_rate": float(anomaly_mask.mean()),
            "mean_anomaly_score": float(scores[anomaly_mask].mean()) if anomaly_mask.any() else 0.0,
        }
        
        # Per-channel analysis
        channel_stats = {}
        for ch in self.FEATURE_CHANNELS:
            if ch in df.columns:
                normal_mean = df.loc[~anomaly_mask, ch].mean()
                anomaly_mean = df.loc[anomaly_mask, ch].mean() if anomaly_mask.any() else normal_mean
                channel_stats[ch] = {
                    "normal_mean": float(normal_mean),
                    "anomaly_mean": float(anomaly_mean),
                    "deviation_pct": float(abs(anomaly_mean - normal_mean) / (abs(normal_mean) + 1e-9) * 100)
                }
        
        report["channel_stats"] = channel_stats
        return report


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from pipeline.ingest import generate_race_telemetry
    
    print("Anomaly Detection Demo")
    df = generate_race_telemetry(n_laps=30, samples_per_lap=500)
    
    detector = TelemetryAnomalyDetector(contamination=0.03)
    detector.fit(df)
    report = detector.get_anomaly_report(df)
    
    print(f"Total frames: {report['total_frames']}")
    print(f"Anomalies: {report['anomaly_count']} ({report['anomaly_rate']:.1%})")
    print("\nTop channels by deviation in anomalies:")
    sorted_channels = sorted(report["channel_stats"].items(),
                              key=lambda x: -x[1]["deviation_pct"])
    for ch, stats in sorted_channels[:5]:
        print(f"  {ch}: {stats['deviation_pct']:.1f}% deviation")
