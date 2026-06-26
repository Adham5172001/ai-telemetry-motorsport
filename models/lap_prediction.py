"""
Lap Time Prediction Model
Author: Adham Aboulkheir | Saudi Motorsport Company
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipeline.features import extract_all_lap_features


class LapTimePredictor:
    """
    Gradient Boosting model for lap time prediction.
    Uses aggregate lap features to predict optimal lap time.
    """
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 5):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=0.05, subsample=0.8, random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
    
    def fit(self, lap_features: pd.DataFrame, lap_times: np.ndarray) -> "LapTimePredictor":
        """Train on lap features and corresponding lap times."""
        self.feature_cols = [c for c in lap_features.columns if c != "lap"]
        X = lap_features[self.feature_cols].fillna(0).values
        X_s = self.scaler.fit_transform(X)
        self.model.fit(X_s, lap_times)
        return self
    
    def predict(self, lap_features: pd.DataFrame) -> np.ndarray:
        X = lap_features[self.feature_cols].fillna(0).values
        X_s = self.scaler.transform(X)
        return self.model.predict(X_s)
    
    def get_feature_importance(self) -> pd.DataFrame:
        importances = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_
        }).sort_values("importance", ascending=False)
        return importances


if __name__ == "__main__":
    from pipeline.ingest import generate_race_telemetry
    
    print("Lap Time Prediction Demo")
    df = generate_race_telemetry(n_laps=50, samples_per_lap=500)
    
    # Generate lap times (base 95s + degradation + noise)
    lap_times = np.array([
        95.2 - 0.05 * lap + 0.3 * np.sin(lap / 5) + np.random.normal(0, 0.4)
        for lap in range(50)
    ])
    
    lap_features = extract_all_lap_features(df)
    
    # Train/test split
    train_features = lap_features.iloc[:40]
    test_features  = lap_features.iloc[40:]
    train_times = lap_times[:40]
    test_times  = lap_times[40:]
    
    predictor = LapTimePredictor()
    predictor.fit(train_features, train_times)
    predictions = predictor.predict(test_features)
    
    mae = mean_absolute_error(test_times, predictions)
    print(f"Lap Time Prediction MAE: {mae:.3f}s")
    print("\nSample predictions:")
    for i, (pred, actual) in enumerate(zip(predictions[:5], test_times[:5])):
        print(f"  Lap {41+i}: Predicted={pred:.2f}s | Actual={actual:.2f}s | Error={abs(pred-actual):.2f}s")
    
    print("\nTop 5 Predictive Features:")
    print(predictor.get_feature_importance().head(5).to_string(index=False))
