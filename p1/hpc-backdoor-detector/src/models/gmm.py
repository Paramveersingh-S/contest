from sklearn.mixture import GaussianMixture
import joblib
import os
import numpy as np

class GMMDetector:
    def __init__(self, max_components: int = 5):
        self.max_components = max_components
        self.model = None
        
    def fit(self, X_train: np.ndarray):
        best_gmm = None
        best_bic = float('inf')
        
        n_comp = min(self.max_components, max(1, len(X_train) // 10))
        
        for k in range(1, n_comp + 1):
            gmm = GaussianMixture(n_components=k, random_state=42)
            gmm.fit(X_train)
            bic = gmm.bic(X_train)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                
        self.model = best_gmm
        
    def score(self, X: np.ndarray) -> np.ndarray:
        # Score is negative log likelihood
        return -self.model.score_samples(X)
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'gmm.pkl'))
        
    def load(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, 'gmm.pkl'))
