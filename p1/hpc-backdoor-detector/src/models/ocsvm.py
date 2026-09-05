from sklearn.svm import OneClassSVM
import joblib
import os
import numpy as np

class OCSVMDetector:
    def __init__(self):
        self.model = OneClassSVM(kernel='rbf', nu=0.05)
        
    def fit(self, X_train: np.ndarray):
        self.model.fit(X_train)
        
    def score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.decision_function(X)
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(model_dir, 'ocsvm.pkl'))
        
    def load(self, model_dir: str):
        self.model = joblib.load(os.path.join(model_dir, 'ocsvm.pkl'))
