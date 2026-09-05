import pandas as pd
from typing import Tuple

def load_raw_traces(filepath: str) -> pd.DataFrame:
    """Load the raw HPC traces CSV."""
    df = pd.read_csv(filepath)
    return df

def clean_traces(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the loaded HPC traces."""
    # Drop rows with NaN
    df = df.dropna()
    # Drop duplicates
    df = df.drop_duplicates()
    
    # Drop zero-variance columns
    variances = df.var()
    zero_var_cols = variances[variances == 0].index
    if len(zero_var_cols) > 0:
        df = df.drop(columns=zero_var_cols)
        
    return df

def split_traces(df: pd.DataFrame, val_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and clean-validation sets."""
    # Random split as we only have clean data without explicit labels
    val_size = int(len(df) * val_frac)
    
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    val_set = shuffled.iloc[:val_size].copy()
    train_set = shuffled.iloc[val_size:].copy()
    
    return train_set, val_set
