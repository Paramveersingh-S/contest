import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
import joblib
import os

def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """Normalize anomaly scores using rank transformation to [0,1]."""
    return rankdata(scores) / len(scores)

class EnsembleDetector:
    def __init__(self):
        self.meta_model = LogisticRegression(random_state=42)
        self.fitted = False
        
    def fit(self, scores_matrix: np.ndarray, labels: np.ndarray):
        """
        scores_matrix: shape (n_samples, n_models)
        labels: shape (n_samples,) binary (0 = clean, 1 = anomaly)
        """
        self.meta_model.fit(scores_matrix, labels)
        self.fitted = True
        
    def predict_proba(self, scores_matrix: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Ensemble not fitted yet.")
        return self.meta_model.predict_proba(scores_matrix)[:, 1]
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.meta_model, os.path.join(model_dir, 'ensemble.pkl'))
        
    def load(self, model_dir: str):
        self.meta_model = joblib.load(os.path.join(model_dir, 'ensemble.pkl'))
        self.fitted = True
