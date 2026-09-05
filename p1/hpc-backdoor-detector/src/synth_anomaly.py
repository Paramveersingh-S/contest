import pandas as pd
import numpy as np

class SyntheticAnomalyGenerator:
    def __init__(self, random_seed: int = 42):
        self.rng = np.random.RandomState(random_seed)

    def fit(self, df_train: pd.DataFrame):
        self.train_std = df_train.std()
        self.train_percentiles = {col: np.percentile(df_train[col], [10, 90]) for col in df_train.columns}
        self.columns = df_train.columns
        self.df_train = df_train.copy()
        
    def generate_anomalies(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """Takes a clean dataframe and applies various corruptions to create anomalies."""
        df_anom = df_clean.copy()
        df_anom['anomaly_type'] = 'none'
        
        n = len(df_anom)
        families = ['subset_shift', 'global_noise', 'distributional_resample', 'correlation_shuffle', 'mixup']
        
        corruption_types = self.rng.choice(families, size=n)
        
        for i in range(n):
            ctype = corruption_types[i]
            row = df_anom.iloc[i].copy()
            
            if ctype == 'subset_shift':
                k = max(1, int(len(self.columns) * self.rng.uniform(0.1, 0.4)))
                cols_to_shift = self.rng.choice(self.columns, size=k, replace=False)
                shift_mag = self.rng.uniform(2, 5, size=k) * self.rng.choice([-1, 1], size=k)
                row[cols_to_shift] += shift_mag * self.train_std[cols_to_shift]
                
            elif ctype == 'global_noise':
                if self.rng.rand() > 0.5:
                    scale = self.rng.uniform(1.15, 1.6)
                else:
                    scale = self.rng.uniform(0.4, 0.85)
                row[self.columns] *= scale
                
            elif ctype == 'distributional_resample':
                k = max(1, int(len(self.columns) * self.rng.uniform(0.1, 0.4)))
                cols = self.rng.choice(self.columns, size=k, replace=False)
                for col in cols:
                    p10, p90 = self.train_percentiles[col]
                    # To avoid bounds issues, we define robust spread
                    spread = max(1e-5, p90 - p10)
                    if row[col] < p10:
                        row[col] = self.rng.uniform(p90, p90 + spread)
                    elif row[col] > p90:
                        row[col] = self.rng.uniform(max(0, p10 - spread), p10)
                    else:
                        if self.rng.rand() > 0.5:
                            row[col] = self.rng.uniform(max(0, p10 - spread), p10)
                        else:
                            row[col] = self.rng.uniform(p90, p90 + spread)
                            
            elif ctype == 'correlation_shuffle':
                k = max(1, int(len(self.columns) * self.rng.uniform(0.1, 0.4)))
                cols = self.rng.choice(self.columns, size=k, replace=False)
                for col in cols:
                    row[col] = self.rng.choice(self.df_train[col].values)
                    
            elif ctype == 'mixup':
                alpha = self.rng.uniform(0.7, 0.9)
                other_idx = self.rng.choice(len(self.df_train))
                other_row = self.df_train.iloc[other_idx]
                row[self.columns] = alpha * row[self.columns] + (1 - alpha) * other_row[self.columns]
                
            row['anomaly_type'] = ctype
            df_anom.iloc[i] = row
            
        return df_anom

def create_synth_val_set(df_clean_val: pd.DataFrame, generator: SyntheticAnomalyGenerator) -> pd.DataFrame:
    df_anom = generator.generate_anomalies(df_clean_val)
    
    df_clean_val = df_clean_val.copy()
    df_clean_val['label'] = 0
    df_clean_val['anomaly_type'] = 'none'
    
    df_anom['label'] = 1
    
    synth_val_set = pd.concat([df_clean_val, df_anom], ignore_index=True)
    return synth_val_set
