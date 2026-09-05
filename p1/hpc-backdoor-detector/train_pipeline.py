import argparse
import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from data import load_raw_traces, clean_traces, split_traces
from synth_anomaly import SyntheticAnomalyGenerator, create_synth_val_set
from features import FeatureBuilder
from models.autoencoder import Autoencoder
from models.gmm import GMMDetector
from models.iforest import IForestDetector
from models.ocsvm import OCSVMDetector
from models.lof import LOFDetector
from ensemble import EnsembleDetector, rank_normalize
from evaluate import evaluate_thresholds

def main():
    print("Starting Training Pipeline...")
    
    # 1. Data loading and splitting
    print("Loading and cleaning data...")
    df = load_raw_traces('../trace.csv')
    df = clean_traces(df)
    train_set, val_set = split_traces(df, val_frac=0.2)
    print(f"Train size: {len(train_set)}, Val size (clean): {len(val_set)}")
    
    # 2. Synthetic anomaly generation
    print("Generating synthetic validation set...")
    synth_gen = SyntheticAnomalyGenerator()
    synth_gen.fit(train_set)
    synth_val_set = create_synth_val_set(val_set, synth_gen)
    print(f"Synthetic Val Set Size: {len(synth_val_set)} (Anomalies: {synth_val_set['label'].sum()})")
    
    # 3. Features
    print("Building features...")
    fb = FeatureBuilder()
    fb.fit(train_set)
    fb.save('models/')
    
    train_feats = fb.transform(train_set)
    val_feats = fb.transform(synth_val_set)
    
    # 4. Base Models
    print("Training base models...")
    models = {
        'ae': Autoencoder(input_dim=train_feats['full'].shape[1]),
        'gmm': GMMDetector(),
        'iforest': IForestDetector(),
        'ocsvm': OCSVMDetector(),
        'lof': LOFDetector()
    }
    
    # Autoencoder uses full features, GMM/OCSVM use reduced
    models['ae'].fit(train_feats['full'])
    models['gmm'].fit(train_feats['reduced'])
    models['iforest'].fit(train_feats['full'])
    models['ocsvm'].fit(train_feats['reduced'])
    models['lof'].fit(train_feats['full'])
    
    for name, model in models.items():
        model.save('models/')
        
    # 5. Ensemble scoring
    print("Training ensemble...")
    # Get scores on train to fit ensemble/normalizer
    def get_all_scores(feats, models_dict):
        scores = {}
        scores['ae'] = models_dict['ae'].score(feats['full'])
        scores['gmm'] = models_dict['gmm'].score(feats['reduced'])
        scores['iforest'] = models_dict['iforest'].score(feats['full'])
        scores['ocsvm'] = models_dict['ocsvm'].score(feats['reduced'])
        scores['lof'] = models_dict['lof'].score(feats['full'])
        return scores
        
    val_raw_scores = get_all_scores(val_feats, models)
    
    # We will use rank normalization but without fitting on train, we can just rank normalize the val set 
    # to train the meta-classifier, or better, we train meta-classifier directly on raw scores 
    # since Logistic Regression can handle scaling.
    val_scores_mat = np.column_stack([val_raw_scores[m] for m in ['ae', 'gmm', 'iforest', 'ocsvm', 'lof']])
    
    ensemble = EnsembleDetector()
    ensemble.fit(val_scores_mat, synth_val_set['label'].values)
    ensemble.save('models/')
    
    # 6. Evaluation and Thresholding
    print("Evaluating and thresholding...")
    ensemble_preds = ensemble.predict_proba(val_scores_mat)
    results = evaluate_thresholds(synth_val_set['label'].values, ensemble_preds, 'models/')
    
    print("\\n--- Pipeline Complete ---")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
        
    # Phase 8: Retrain on all data
    print("\\nRetraining on all available clean data...")
    fb_all = FeatureBuilder()
    fb_all.fit(df)
    fb_all.save('models/')
    all_feats = fb_all.transform(df)
    
    models['ae'].fit(all_feats['full'])
    models['gmm'].fit(all_feats['reduced'])
    models['iforest'].fit(all_feats['full'])
    models['ocsvm'].fit(all_feats['reduced'])
    models['lof'].fit(all_feats['full'])
    
    for name, model in models.items():
        model.save('models/')
        
    print("Retraining completed and artifacts saved to models/")

if __name__ == '__main__':
    main()
