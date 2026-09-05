import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
import joblib

def evaluate_thresholds(y_true: np.ndarray, y_scores: np.ndarray, model_dir: str):
    """Evaluate over synthetic val set and find best F1 threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    best_f1 = 0
    best_threshold = 0
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            
    # Calculate metrics at best threshold
    y_pred_best = (y_scores >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_best).ravel()
    
    tpr_best = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr_best = fp / (fp + tn) if (fp + tn) > 0 else 0
    acc_best = (tp + tn) / (tp + tn + fp + fn)
    
    results = {
        'auroc': roc_auc,
        'best_threshold': best_threshold,
        'f1': best_f1,
        'tpr': tpr_best,
        'fpr': fpr_best,
        'accuracy': acc_best
    }
    
    # Save threshold
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(best_threshold, os.path.join(model_dir, 'threshold.pkl'))
    
    # Plot ROC
    os.makedirs('report', exist_ok=True)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter([fpr_best], [tpr_best], color='red', marker='x', s=100, label=f'Best F1 Threshold')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('report/roc_curve.png')
    
    return results
