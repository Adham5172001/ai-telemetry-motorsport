"""
AI Telemetry Analysis — Full Demo
Author: Adham Aboulkheir | Saudi Motorsport Company
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from pipeline.ingest import generate_race_telemetry
from pipeline.features import extract_all_lap_features
from models.anomaly import TelemetryAnomalyDetector
from models.lap_prediction import LapTimePredictor


def main():
    print("=" * 60)
    print("AI TELEMETRY ANALYSIS — MOTORSPORT PERFORMANCE")
    print("Author: Adham Aboulkheir | Saudi Motorsport Company")
    print("=" * 60)
    
    os.makedirs("outputs", exist_ok=True)
    
    print("\n[1/4] Generating race telemetry data...")
    df = generate_race_telemetry(n_laps=50, samples_per_lap=500)
    print(f"  {len(df)} frames | {df['lap'].nunique()} laps | {len(df.columns)} channels")
    
    print("\n[2/4] Anomaly detection...")
    detector = TelemetryAnomalyDetector(contamination=0.03)
    detector.fit(df)
    report = detector.get_anomaly_report(df)
    print(f"  Anomalies: {report['anomaly_count']} / {report['total_frames']} ({report['anomaly_rate']:.1%})")
    
    print("\n[3/4] Lap time prediction...")
    lap_times = np.array([
        95.2 - 0.05 * lap + 0.3 * np.sin(lap / 5) + np.random.normal(0, 0.4)
        for lap in range(50)
    ])
    lap_features = extract_all_lap_features(df)
    predictor = LapTimePredictor()
    predictor.fit(lap_features.iloc[:40], lap_times[:40])
    predictions = predictor.predict(lap_features.iloc[40:])
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(lap_times[40:], predictions)
    print(f"  Lap Time MAE: {mae:.3f}s")
    
    print("\n[4/4] Generating visualisations...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
    
    # Lap times
    laps = np.arange(1, 51)
    axes[0].plot(laps, lap_times, color="#58a6ff", linewidth=1.5, label="Actual", alpha=0.8)
    pred_all = predictor.predict(lap_features)
    axes[0].plot(laps, pred_all, color="#00c9b1", linewidth=2, linestyle="--", label="Predicted")
    axes[0].set_title("Lap Time Prediction", color="white")
    axes[0].set_xlabel("Lap", color="white")
    axes[0].set_ylabel("Lap Time (s)", color="white")
    axes[0].legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    axes[0].tick_params(colors="white")
    axes[0].grid(alpha=0.3, color="#21262d")
    
    # Tyre degradation
    tyre_means = [df[df["lap"] == l]["tyre_temp_fl"].mean() for l in range(1, 51)]
    axes[1].plot(laps, tyre_means, color="#f4a261", linewidth=2)
    axes[1].fill_between(laps, tyre_means, alpha=0.2, color="#f4a261")
    axes[1].set_title("Tyre Temperature Trend", color="white")
    axes[1].set_xlabel("Lap", color="white")
    axes[1].set_ylabel("Tyre Temp (°C)", color="white")
    axes[1].tick_params(colors="white")
    axes[1].grid(alpha=0.3, color="#21262d")
    
    # Anomaly scores
    scores = detector.anomaly_scores(df)
    axes[2].plot(scores[:2000], color="#00c9b1", linewidth=0.5, alpha=0.7)
    axes[2].axhline(y=np.percentile(scores, 5), color="#ff7b72", linestyle="--",
                    linewidth=1.5, label="Anomaly threshold")
    axes[2].set_title("Anomaly Scores", color="white")
    axes[2].set_xlabel("Frame", color="white")
    axes[2].set_ylabel("Score (lower=anomalous)", color="white")
    axes[2].legend(facecolor="#161b22", labelcolor="white", fontsize=8)
    axes[2].tick_params(colors="white")
    axes[2].grid(alpha=0.3, color="#21262d")
    
    plt.tight_layout()
    plt.savefig("outputs/motorsport_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("  Saved: outputs/motorsport_results.png")
    
    print("\n✓ Analysis complete!")
    print(f"  Anomaly rate: {report['anomaly_rate']:.1%}")
    print(f"  Lap time MAE: {mae:.3f}s")


if __name__ == "__main__":
    main()
