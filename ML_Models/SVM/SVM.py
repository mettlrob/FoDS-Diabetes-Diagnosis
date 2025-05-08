#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('../../data/cleaned_diabetes.csv')

X = data.drop(columns = ['Outcome'])
y = data['Outcome']


"""define hyperparameter grid"""
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto'] #relevant for rbf
}

""""Set up outer CV"""
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

outer_scores = []
outer_fold = 1

#To store predictions for later inspection

all_conf_matrices = []
all_reports = []
roc_auc_list = []

f1_list = []
precision_list = []
recall_list = []

""""Outer CV loop"""
for train_idx, test_idx in outer_cv.split(X, y):

    print(f"\n--- Outer Fold {outer_fold} ---")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    """"Inner CV for hyperparameter tuning"""

    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator = SVC(probability= True, class_weight = 'balanced', random_state=42),
        param_grid = param_grid,
        cv = inner_cv,
        scoring = 'roc_auc',
        n_jobs = -1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")

    """Evaluate best model on outer test set"""

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    # y_pred_proba = best_model.decision_function(X_test) #unprobabilistic scores

    #ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    all_conf_matrices.append(confusion_matrix(y_test, y_pred))
    roc_auc_list.append(roc_auc)
    #Classification report
    report = classification_report(y_test, y_pred, output_dict = True)
    all_reports.append(report)

    #collect metrics for postive class (diabetes = 1)
    f1_list.append(report['1']['f1-score'])
    precision_list.append(report['1']['precision'])
    recall_list.append(report['1']['recall'])

    """"Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Fold {outer_fold} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Fold{outer_fold}')
    plt.legend()
    roc_path = f'../../output/SVM_output/ROC/ROC_Fold{outer_fold}.png'

    plt.savefig(roc_path)
    plt.close()

    outer_fold += 1

    """Summary of outer fold ROC AUCs"""

    print("\n ==== Outer Fold ROC AUC Scores ===")
    for i, auc in enumerate(roc_auc_list, 1):
        print(f"Fold {i}: {auc:.3f}")

    print(f"\nMean ROC AUC: {np.mean(roc_auc_list):.3f}")
    print(f"Std ROC AUC: {np.std(roc_auc_list):.3f}")

    """Optional: Average confusion matrix """

    cm = confusion_matrix(y_test, y_pred) #recompute confusion matrix for the current fold

    cm_percent = cm/ cm.sum(axis = 1, keepdims = True) *100 

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot = True, fmt = '.2f', cmap = 'Blues', cbar = False, 
                xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
    plt.title('Confusion Matrix Fold {outer_fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f'../../output/SVM_output/Confusion_Matrix/CM_Fold{outer_fold}.png'
    plt.savefig(cm_path)
    plt.close()

#plot mean Coonfusion matrix across all outer folds
mean_cm = np.mean(all_conf_matrices, axis = 0)
mean_cm_percent = mean_cm/ mean_cm.sum(axis = 1, keepdims = True) *100
plt.figure(figsize=(6,5))
sns.heatmap(mean_cm_percent, annot = True, fmt = '.2f', cmap = 'Blues', cbar = False, 
            xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
plt.title('Mean Confusion Matrix (%) across Outer Folds')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm__mean_path = f'../../output/SVM_output/Mean_CM.png'
plt.savefig(cm__mean_path)
plt.close()

print(f'Mean ROC AUC: {np.mean(roc_auc_list):.3f}')
print(f'Std ROC AUC: {np.std(roc_auc_list):.3f}')
print(f'Mean F1 Score: {np.mean(f1_list):.3f}')
print(f'Mean Precision: {np.mean(precision_list):.3f}')
print(f'Mean Recall: {np.mean(recall_list):.3f}')




