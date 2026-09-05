import argparse
import sys
import os
import pandas as pd
import numpy as np
import joblib

from data import clean_traces
from features import FeatureBuilder
from models.autoencoder import Autoencoder
from models.gmm import GMMDetector
from models.iforest import IForestDetector
from models.ocsvm import OCSVMDetector
from models.lof import LOFDetector
from ensemble import EnsembleDetector

def predict(csv_path: str, output_path: str = 'predictions.csv'):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    # clean but keep trace IDs if any? 
    # For now we just clean NaNs
    df_clean = clean_traces(df)
    
    print("Loading models...")
    model_dir = os.path.join(os.path.dirname(__file__), '../models')
    
    fb = FeatureBuilder()
    fb.load(model_dir)
    feats = fb.transform(df_clean)
    
    ae = Autoencoder(input_dim=feats['full'].shape[1])
    ae.load(model_dir)
    gmm = GMMDetector()
    gmm.load(model_dir)
    iforest = IForestDetector()
    iforest.load(model_dir)
    ocsvm = OCSVMDetector()
    ocsvm.load(model_dir)
    lof = LOFDetector()
    lof.load(model_dir)
    
    scores_ae = ae.score(feats['full'])
    scores_gmm = gmm.score(feats['reduced'])
    scores_iforest = iforest.score(feats['full'])
    scores_ocsvm = ocsvm.score(feats['reduced'])
    scores_lof = lof.score(feats['full'])
    
    scores_mat = np.column_stack([scores_ae, scores_gmm, scores_iforest, scores_ocsvm, scores_lof])
    
    ensemble = EnsembleDetector()
    ensemble.load(model_dir)
    final_probs = ensemble.predict_proba(scores_mat)
    
    threshold = joblib.load(os.path.join(model_dir, 'threshold.pkl'))
    
    labels = (final_probs >= threshold).astype(int)
    
    results = pd.DataFrame({
        'trace_id': df_clean.index,
        'anomaly_score': final_probs,
        'label': labels
    })
    
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Detected {labels.sum()} anomalies out of {len(labels)} traces.")
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run HPC Backdoor Detector Inference")
    parser.add_argument('--csv_path', type=str, required=True, help="Path to HPC trace CSV file")
    parser.add_argument('--output', type=str, default='predictions.csv', help="Output CSV path")
    args = parser.parse_args()
    
    predict(args.csv_path, args.output)
