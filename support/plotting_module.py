import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix

def plot_roc_curves(results, X, y, splits, outdir = '../../output/pipeline_output/roc/'):
    os.makedirs(outdir, exist_ok=True)
    for name, cv_results in results.items():
        plt.figure(figsize=(6,6))
        for i, (train_idx, test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i]
            best_model = gs.best_estimator_
            y_score = best_model.predict_proba(X[test_idx])[:,1]
            fpr, tpr, _ = roc_curve(y[test_idx], y_score)
            plt.plot(fpr, tpr, label=f"Fold {i+1}")
        plt.plot([0,1],[0,1], 'k--', alpha=0.3)
        plt.title(f"{name} ROC Curve (outer CV)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_roc.png")
        plt.close()

def plot_confusion_matrices(results, X, y, splits, outdir = '../../output/pipeline_output/cm/'):
    os.makedirs(outdir, exist_ok=True)
    for name, cv_results in results.items():
        total_cm = np.zeros((2,2), dtype=float)
        total_samples = 0
        for i, (train_idx, test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i]
            best_model = gs.best_estimator_
            y_pred = best_model.predict(X[test_idx])
            cm = confusion_matrix(y[test_idx], y_pred)
            total_cm += cm
            total_samples += len(test_idx)
        # convert to percentage
        cm_perc = total_cm / total_samples * 100
        plt.figure(figsize=(4,4))
        sns.heatmap(cm_perc, annot=True, fmt=".1f", cmap="Blues",
                    xticklabels=["neg","pos"], yticklabels=["neg","pos"])
        plt.title(f"{name} Avg. Confusion Matrix (%)")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_cm.png")
        plt.close()