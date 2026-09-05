from sklearn.ensemble import IsolationForest
import joblib
import os
import numpy as np

class IForestDetector:
    def __init__(self):
        self.model = IsolationForest(random_state=42)
        
    def fit(self, X_train: np.ndarray):
        self.model.fit(X_train)
        
    def score(self, X: np.ndarray) -> np.ndarray:
        # Score = -decision_function (higher = more anomalous)
        return -self.model.decision_function(X)
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'iforest.pkl'))
        
    def load(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, 'iforest.pkl'))
