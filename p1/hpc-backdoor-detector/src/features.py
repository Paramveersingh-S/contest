import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

class FeatureBuilder:
    def __init__(self, n_components: int = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.fitted = False
        self.ratio_features = []
        self.base_columns = []
        
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Find likely ratio candidates if present
        # e.g., cache-misses / cache-references
        # Our CSV has: cache-references, cycles, LLC-loads
        if 'cache-references' in df.columns and 'LLC-loads' in df.columns:
            # We can create a ratio feature here
            df['llc_to_cache_ratio'] = df['LLC-loads'] / (df['cache-references'] + 1e-9)
            if 'llc_to_cache_ratio' not in self.ratio_features:
                self.ratio_features.append('llc_to_cache_ratio')
                
        if 'cache-references' in df.columns and 'cycles' in df.columns:
            df['cache_per_cycle'] = df['cache-references'] / (df['cycles'] + 1e-9)
            if 'cache_per_cycle' not in self.ratio_features:
                self.ratio_features.append('cache_per_cycle')
                
        return df

    def fit(self, df_train: pd.DataFrame):
        df = self._add_ratio_features(df_train)
        self.base_columns = df.columns.tolist()
        
        self.scaler.fit(df)
        X_scaled = self.scaler.transform(df)
        
        # Determine n_components dynamically if base_columns is small
        n_comp = min(2, len(self.base_columns))
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(X_scaled)
        
        self.fitted = True
        
    def transform(self, df: pd.DataFrame) -> dict:
        """Returns both full features and PCA reduced features"""
        if not self.fitted:
            raise ValueError("FeatureBuilder not fitted yet.")
            
        df = self._add_ratio_features(df)
        
        # Ensure columns match train
        for col in self.base_columns:
            if col not in df.columns:
                df[col] = 0.0
                
        df = df[self.base_columns]
        
        X_scaled = self.scaler.transform(df)
        X_pca = self.pca.transform(X_scaled)
        
        return {
            'full': X_scaled,
            'reduced': X_pca
        }
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        joblib.dump(self.pca, os.path.join(model_dir, 'pca.pkl'))
        
    def load(self, model_dir: str):
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.pca = joblib.load(os.path.join(model_dir, 'pca.pkl'))
        self.fitted = True
