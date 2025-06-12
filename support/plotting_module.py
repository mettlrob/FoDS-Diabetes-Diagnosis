import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, precision_recall_curve, average_precision_score



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


def plot_pr_curves(results, X, y, splits, outdir = '../pipeline_output/pr/'):
    """
    Plot Precision-Recall curves per model with one curve per outer CV fold. 
    Labels include average precision (AP) per fold.

    """
    os.makedirs(outdir, exist_ok=True)
    for name, cv_results in results.items():
        plt.figure(figsize= (6,6))
        for i, (_, test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i]
            best_model = gs.best_estimator_
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            y_score = best_model.predict_proba(X_test)[:,1]
            precision, recall, _ = precision_recall_curve(y_test, y_score)
            ap = average_precision_score(y_test, y_score)
            plt.step(recall, precision, where = 'post', label=f"Fold {i+1} (AP={ap:.3f})")
        #no-skill line: positive rate
        pos_rate = np.mean(y == 1)
        plt.hlines(pos_rate, 0, 1, colors = 'k', linestyles='--', label='No Skill')
        plt.xlabel("Recall (Sensitivity)")
        plt.ylabel("Precision")
        plt.title(f"{name} Precision-Recall (outer CV)")
        plt.legend(loc ="lower left")
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_pr.png", dpi = 300)
        plt.close()

def plot_metric_comparison(results, outdir = '../pipeline_output/metric_comparison/'):
    """
    Barplot comparing mean test scores (with error bars) across models for each metric in 
    [accuracy, precision, recall, f1, roc_auc].
    
    """
    os.makedirs(outdir, exist_ok= True)

    rows = []
    for name, cv_results in results.items():
        stats = {'model': name}
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            vals = cv_results[f'test_{metric}']
            stats[f'{metric}_mean'] = np.mean(vals)
            stats[f'{metric}_std'] = np.std(vals)
        rows.append(stats)
    df = pd.DataFrame(rows)
    # Map raw names to pretty labels
    metric_map = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'roc_auc': 'ROC AUC'
    }
    model_map = {
        'LogisticRegression': 'Logistic Regression',
        'SVC':                'SVM',
        'KNN':                'K-Nearest Neighbors',
        'RandomForest':       'Random Forest'
    }

    # Melt into long form
    data = []
    for _, row in df.iterrows():
        for raw in ['accuracy','precision','recall','f1','roc_auc']:
            data.append({
                'model':       model_map.get(row['model'], row['model']),
                'metric':      metric_map[raw],
                'mean':        row[f'{raw}_mean'],
                'std':         row[f'{raw}_std']
            })
    plot_df = pd.DataFrame(data)

    plt.figure(figsize=(10,6))
    ax = sns.barplot(
        data=plot_df,
        x='metric',
        y='mean',
        hue='model',
        palette='pastel',
        ci=None,
        capsize=0.1,
        errcolor=None # disable seaborn's own error bars
    )  
    # Manually add error bars per bar
    # Each bar is a Patch in ax.patches
    # Grouped by metric categories
    # Each bar is a Rectangle patch; zip it with the corresponding std

    for patch, std in zip(ax.patches, plot_df['std']):
        x = patch.get_x() + patch.get_width() / 2
        height = patch.get_height()
        ax.errorbar(x, height, yerr=std, ecolor='black', capsize=5, fmt='none')

    plt.xlabel('Metric')
    plt.ylabel('Test Score')
    plt.ylim(0,1)
    plt.title('Model Comparison by Metric')
    plt.legend(title='Model', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{outdir}/metrics_comparison.png", dpi=300)
    plt.close()

