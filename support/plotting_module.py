import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def plot_roc_curves(results, X, y, splits, outdir = '../pipeline_output/roc/'):
    os.makedirs(outdir, exist_ok=True)
    for name, cv_results in results.items():
        plt.figure(figsize=(6,6))
        for i, (_ , test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i]
            best_model = gs.best_estimator_
            y_score = best_model.predict_proba(X.iloc[test_idx])[:,1]
            fpr, tpr, _ = roc_curve(y.iloc[test_idx], y_score)
            auc = roc_auc_score(y.iloc[test_idx], y_score)
            plt.plot(fpr, tpr, label=f"Fold {i+1} (AUC={auc:.3f})")
        plt.plot([0,1],[0,1], 'k--', alpha=0.3)
        plt.title(f"{name} ROC Curve (outer CV)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_roc.png", dpi = 300)
        plt.close()

def plot_confusion_matrices(results, X, y, splits, outdir = '../pipeline_output/cm/'):
    os.makedirs(outdir, exist_ok=True)
    n_folds = len(splits)

    for name, cv_results in results.items():
        #accumulator for the percent-matrix of each fold
        sum_pct_cm = np.zeros((2,2), dtype = float)
        for i, (_, test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i]
            best_model = gs.best_estimator_ 

            #get true vs- pred on this fold's test set
            y_true = y.iloc[test_idx]
            y_pred = best_model.predict(X.iloc[test_idx])

            #raw counts
            cm = confusion_matrix(y_true, y_pred).astype(float)

            # normalize *this* cm to percentage *of that fold's total*:
            cm_pct = cm/ cm.sum(axis=1, keepdims=True) * 100
            sum_pct_cm += cm_pct

        #average across folds
        avg_cm = sum_pct_cm / n_folds
        plt.figure(figsize=(6,5))
        sns.heatmap(avg_cm, annot=True, fmt=".1f", cmap="Blues", cbar = True,
                    xticklabels=["No Diabetes","Diabetes"], yticklabels=["No Diabetes","Diabetes"])
        plt.title(f"{name} Avg. Confusion Matrix (%)")
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.savefig(f"{outdir}/{name}_cm.png", dpi = 300)
        plt.close()