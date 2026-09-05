import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data import load_raw_traces, clean_traces, split_traces
from synth_anomaly import SyntheticAnomalyGenerator, create_synth_val_set
from features import FeatureBuilder
from models.autoencoder import Autoencoder
from models.gmm import GMMDetector
from models.iforest import IForestDetector
from models.ocsvm import OCSVMDetector
from models.lof import LOFDetector

def main():
    print("Running Ablation Study...")
    
    # 1. Setup
    df = load_raw_traces('../trace.csv')
    df = clean_traces(df)
    train_set, val_set = split_traces(df, val_frac=0.2, seed=42)
    
    synth_gen = SyntheticAnomalyGenerator(random_seed=42)
    synth_gen.fit(train_set)
    synth_val_set = create_synth_val_set(val_set, synth_gen)
    y_true = synth_val_set['label'].values
    
    fb = FeatureBuilder()
    fb.fit(train_set)
    
    train_feats = fb.transform(train_set)
    val_feats = fb.transform(synth_val_set)
    
    # 2. Train Models
    models = {
        'Autoencoder': Autoencoder(input_dim=train_feats['full'].shape[1]),
        'GMM': GMMDetector(),
        'IForest': IForestDetector(),
        'OCSVM': OCSVMDetector(),
        'LOF': LOFDetector()
    }
    
    models['Autoencoder'].fit(train_feats['full'])
    models['GMM'].fit(train_feats['reduced'])
    models['IForest'].fit(train_feats['full'])
    models['OCSVM'].fit(train_feats['reduced'])
    models['LOF'].fit(train_feats['full'])
    
    # 3. Score
    scores = {}
    scores['Autoencoder'] = models['Autoencoder'].score(val_feats['full'])
    scores['GMM'] = models['GMM'].score(val_feats['reduced'])
    scores['IForest'] = models['IForest'].score(val_feats['full'])
    scores['OCSVM'] = models['OCSVM'].score(val_feats['reduced'])
    scores['LOF'] = models['LOF'].score(val_feats['full'])
    
    print("\n--- Ablation AUROC Table ---")
    for name, s in scores.items():
        # normalize to avoid nan issues, though auroc doesn't care about scale
        try:
            auc = roc_auc_score(y_true, s)
            print(f"{name}: {auc:.4f}")
        except:
            print(f"{name}: error")
            
    # Meta ensemble score
    from ensemble import EnsembleDetector
    scores_mat = np.column_stack([scores['Autoencoder'], scores['GMM'], scores['IForest'], scores['OCSVM'], scores['LOF']])
    ensemble = EnsembleDetector()
    ensemble.fit(scores_mat, y_true)
    y_pred_ens = ensemble.predict_proba(scores_mat)
    print(f"Meta-Ensemble: {roc_auc_score(y_true, y_pred_ens):.4f}")

if __name__ == '__main__':
    main()
