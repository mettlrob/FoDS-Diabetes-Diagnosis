#%%
import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import warnings
warnings.filterwarnings("ignore")


#create helper function to plot ROC curves in one figure
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes



data = pd.read_csv('../../Data_Processing/no_transformer_data.csv')

X = data.drop(columns = ['Outcome'])
y = data['Outcome']


"""define hyperparameter grid"""
param_grid = [
    { 'C': [0.1, 1, 10], 'kernel' : ['linear']},
    # { 'C': [0.1, 1, 10], 'kernel' : ['rbf'], 'gamma' : ['scale', 'auto']},
    # { 'C': [0.1, 1, 10], 'kernel' : ['poly'], 'degree' : [2, 3], 'gamma' : ['scale', 'auto']}
]
    
    # 'kernel': ['linear'],
    # #'kernel': ['linear','rbf', 'poly'],
    # 'gamma': ['scale', 'auto'] #relevant for rbf


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

fpr_list = []
tpr_list = []

coef_list = [] #this is for feature importance.

# shap lists
shap_values_all = []
shap_test_sets = []
shap_models = []

scoring = { 
    'f1' : make_scorer(f1_score, average = 'binary', pos_label = 1),
    'recall' : make_scorer(recall_score, pos_label = 1),
    'precision' : make_scorer(precision_score, pos_label = 1),
    'roc_auc' : make_scorer(roc_auc_score, needs_proba = True)

}
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
        scoring = scoring, #for overview and interpretation.
        refit = 'recall', #choose the best model based on recall score
        n_jobs = -1,
        return_train_score = True
    )


    grid_search.fit(X_train, y_train)





    print(f"Best parameters: {grid_search.best_params_}")

    results = pd.DataFrame(grid_search.cv_results_)

    summary_results = results[[
        'param_C',
        'param_kernel',
        'mean_test_f1',
        'mean_test_recall',
        'mean_test_precision',
        'std_test_f1',
        'std_test_recall',
        'std_test_precision',
    ]]

    summary_sorted = summary_results.sort_values(by = 'mean_test_f1', ascending = False)
    print("\n--- Inner CV Results (sorted by F1) ---")
    print(summary_sorted.head())

    """--- Evaluate best model on outer test set ---"""

    best_model = grid_search.best_estimator_
    
    coef_list.append(best_model.coef_.flatten()) #for feature importance

    
    shap_test_sets.append(X_test.copy())
    shap_models.append(best_model) #for shap analysis

    
    explainer = shap.LinearExplainer(best_model, X_train, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test)
    shap_values_all.append(shap_values)

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

    """"--- Plot ROC curve ---"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    if fpr[-1] < 1.0 or tpr[-1] < 1.0:
        fpr = np.append(fpr, 1.0)
        tpr = np.append(tpr, 1.0)
    # Add the identity line
    fpr_list.append(fpr) #to collect all fpr values for plotting in one figure below.
    tpr_list.append(tpr) # to collect all tpr values for plotting in one figure below.
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Fold {outer_fold} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve Fold {outer_fold}')
    plt.legend()
    roc_path = f'../../output/SVM_output/ROC/ROC_Fold_{outer_fold}.png'

    plt.savefig(roc_path, dpi = 300)
    plt.close()

    #outer_fold += 1




    """--- Summary of outer fold ROC AUCs ---"""

    print("\n ==== Outer Fold ROC AUC Scores ===")
    for i, auc in enumerate(roc_auc_list, 1):
        print(f"Fold {i}: {auc:.3f}")

    print(f"\nMean ROC AUC: {np.mean(roc_auc_list):.3f}")
    print(f"Std ROC AUC: {np.std(roc_auc_list):.3f}")
    print("")

    """--- Optional: Average confusion matrix ---"""

    cm = confusion_matrix(y_test, y_pred) #recompute confusion matrix for the current fold

    cm_percent = cm/ cm.sum(axis = 1, keepdims = True) *100 

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_percent, annot = True, fmt = '.2f', cmap = 'Blues', cbar = False, 
                xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
    plt.title(f'Confusion Matrix Fold {outer_fold}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f'../../output/SVM_output/CM/CM_Fold_{outer_fold}.png'
    plt.savefig(cm_path)
    plt.close()
    outer_fold += 1 
    
#plot ROC curves for all outer folds in one figure.
plt.style.use('seaborn-v0_8-darkgrid')
plt.figure(figsize=(8, 6))
ax = plt.gca()
for i in range(len(fpr_list)):
    ax.plot(fpr_list[i], tpr_list[i], label = f'Fold {i+1} AUC = {roc_auc_list[i]:.2f}')

add_identity(ax, color = 'r', ls = '--', label = 'Random Classifier')

ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - SVM (5-fold CV)', fontsize = 14)
ax.legend(loc = 'lower right', fontsize = 10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('../../output/SVM_output/All_ROC_SVM.png', dpi = 300)
plt.close()

#plot mean Coonfusion matrix across all outer folds
mean_cm = np.mean(all_conf_matrices, axis = 0)
mean_cm_percent = mean_cm/ mean_cm.sum(axis = 1, keepdims = True) *100
plt.figure(figsize=(6,5))
sns.heatmap(mean_cm_percent, annot = True, fmt = '.1f', cmap = 'Blues', cbar = True, 
            xticklabels = ['No Diabetes', 'Diabetes'], yticklabels = ['No Diabetes', 'Diabetes'])
plt.title('Mean Confusion Matrix (%; SVM)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
cm__mean_path = f'../../output/SVM_output/Mean_CM.png'
plt.savefig(cm__mean_path, dpi = 300)
plt.close()

print(f'ROC AUC: {np.mean(roc_auc_list):.3f} ± {np.std(roc_auc_list):.3f}')
print(f'F1 Score: {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}')
print(f'Precision: {np.mean(precision_list):.3f} ± {np.std(precision_list):.3f}')
print(f'Recall: {np.mean(recall_list):.3f} ± {np.std(recall_list):.3f}')
print(f'Accuracy: {np.mean([report["accuracy"] for report in all_reports]):.3f} ± {np.std([report["accuracy"] for report in all_reports]):.3f}')


"""The best results were obtained using a linear kernel instead of rbf or poly. This suggest that the data is enough linearly separable.
that means that there is a hyperplane that can separate the two classes well enough without needing curved
or flexible boundaries. the relationships between the features and the class label are additive and not heavily interacting."""

"""--- Feature Importance ---"""


coef_array = np.array(coef_list) #shape: (n_folds, n_features)
avg_coefs = np.mean(coef_array, axis = 0) #average across folds

importance_df = pd.DataFrame({
    'Feature' : X.columns,
    'AvgWeight' : avg_coefs,
    'AbsAvgWeight' : np.abs(avg_coefs)

}).sort_values(by = 'AbsAvgWeight', ascending = False)

# importance_df.to_csv('../../output/SVM_output/Feature_Importance.csv', index = False)

#Plot all 8 features in descending order of importance

plt.figure(figsize=(10, 6))
sns.barplot(data = importance_df.head(10), x = 'AbsAvgWeight', y = 'Feature', palette = 'viridis')
plt.title('Ordered Averaged Feature Importances from Linear SVM (across Folds)')
plt.xlabel('Absolute Averaged Coefficient Weight')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../../output/SVM_output/SVM_feature_importance.png')
plt.close()
"""Linear SVM makes predictions using a weighted sum of the input features. The learned coefficients (wi) 
directly represent the influence of each feature on the decision boundary. Larger absolute values mean 
stronger influence; positive weights push toward the positive class (diabetes), negative toward teh negative class."""


""" --- SHAP Barplot --- """
shap_mean_values_all = []
feature_names = X.columns
for i, (model, X_test_fold) in enumerate(zip(shap_models, shap_test_sets)): 
    explainer = shap.LinearExplainer(model, X_test_fold, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_test_fold)

    #shap_values shape: (n_samples. n_features)
    mean_abs_shap = np.abs(shap_values).mean(axis = 0)
    shap_mean_values_all.append(mean_abs_shap)


#Average SHAP values across folds
shap_values_array= np.array(shap_mean_values_all)
avg_shap = shap_values_array.mean(axis = 0)

#Summary of shap
shap_df = pd.DataFrame({
    'Feature' : feature_names,
    'Mean_SHAP_value' : avg_shap,

}).sort_values(by = 'Mean_SHAP_value', ascending = False)

#plot SHAP feature importance using barplot
plt.figure(figsize=(10, 6))
sns.barplot(data = shap_df.head(10), x = 'Mean_SHAP_value', y = 'Feature', palette = 'inferno')
plt.title('Mean SHAP Feature Importances (Linear SVM across Folds)')
plt.xlabel('Mean SHAP Value')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('../../output/SVM_output/SVM_SHAP_feature_importance.png')
plt.close()

#plot SHAP summary plot
all_shap_values = np.vstack(shap_values_all)
all_X_test = pd.concat(shap_test_sets, axis = 0)

all_X_test.columns = feature_names

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(10, 8))
shap.summary_plot(all_shap_values,
                    all_X_test,
                    feature_names = feature_names,
                    plot_type = 'dot',
                    color_bar_label = 'Feature Value',
                    show = False)

plt.title('Linear SVM - SHAP Summary Plot\n(Aggregated Across Outer Folds)', fontsize = 14, fontweight = 'bold', pad = 20)
plt.xlabel('SHAP Value (Impact on Predicting Diabetes)', fontsize = 12)
plt.ylabel('Feature', fontsize = 12)

plt.tight_layout()
plt.savefig('../../output/SVM_output/SVM_SHAP_summary.png', dpi = 300)
plt.close()


""" --- Collecting and saving useful metrics --- """

summary = {
    'Model' : 'SVM',
    'F1_Mean' : [np.mean(f1_list)],
    'F1_Std' : [np.std(f1_list)],
    'Recall_Mean' : [np.mean(recall_list)],
    'Recall_Std' : [np.std(recall_list)],
    'Precision_Mean' : [np.mean(precision_list)],
    'Precision_Std' : [np.std(precision_list)],
    'Accuracy_Mean' : [np.mean([report["accuracy"] for report in all_reports])],
    'Accuracy_Std' : [np.std([report["accuracy"] for report in all_reports])],
    'ROC_AUC_Mean' : [np.mean(roc_auc_list)],
    'ROC_AUC_Std' : [np.std(roc_auc_list)],

}
df_summary = pd.DataFrame(summary)
df_summary.to_csv('../../Data_Processing/svm_summary_metrics.csv', index = False)




