#%%




# Bibliotheken importieren
import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import (make_scorer, 
                             precision_score, 
                             recall_score, 
                             precision_recall_curve,
                             average_precision_score,
                            # PrecisionRecallDisplay,
                             f1_score, 
                             accuracy_score, 
                             roc_auc_score, 
                             roc_curve, 
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns


"""Dateien Laden und Spliten"""
# Daten laden
#df = pd.read_csv('../../Data_Processing/whole_cleaned_dataset.csv')
df = pd.read_csv('../../Data_Processing/no_transformer_data.csv')

# Features und Zielspalte
X = df.drop('Outcome', axis=1)
y = df['Outcome']



"""Outer und Inner Loops der CV definieren"""
# Stratified K-Fold und K-Fold einrichten
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)


"""Parameter Grid für Inner CV definieren"""
param_grid = {
    'C':      [0.01, 0.1, 1, 10, 100],   # Best Result = 1  ;  Später war 0.1 Bestes Resultat
    #'C':    [0.8, 0.9, 1, 1.1, 1,2],     # Best Result = 0.8
    #'C':    [0.5, 0.6, 0.7, 0.8, 0.9],  # Best Result = 0.8
    #'C':    [0.75, 0.775, 0.8, 0.825, 0.85],    # Best Result = 0.825
    #'C':    [0.05, 0.1, 0.15, 0.2, 0.25],        # Best Result = 0.1
    'penalty':['l1', 'l2'],
    'solver': ['liblinear'],  # liblinear unterstützt l1 und l2
}



# Metriken speichern
precision_list = []
recall_list = []
specificity_list = []
accuracy_list = []
f1_list = []
auc_list = []

# Metriken für ROC-Kurve
mean_fpr = np.linspace(0, 1, 100)
tprs = []
conf_matrices = []

# Metriken für Shap Plot
all_shap_vals = []
all_X_tests  = []

# Normal Feature Importance
coefs_m = []
all_X_trains = []

# Metriken für PR-Kurve
mean_recall = np.linspace(0, 1, 200)
tprs_interp = []      # hier speichern wir für jeden Fold die Precision interpoliert auf mean_recall
aps = []              # Average Precision pro Fold



"""Cross Validation Outer und Inner Loop"""
# Cross-Validation Schleife
for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    
    
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    """
    print("-------------------------------------------------------------------")
    print(outer_cv)
    print("-------------------------------------------------------------------")
    """
    print("-------------------------------------------------------------------")
    print("-------------------------------------------------------------------")


    """Hyperparameter Tuning"""
    # GridSearchCV mit AUC / Recall als Optimierungsziel
    grid = GridSearchCV(
    estimator = LogisticRegression(class_weight='balanced', max_iter=1000),
    param_grid = param_grid,
    cv         = inner_cv,
    #scoring    = 'roc_auc',
    scoring    = 'recall',
    n_jobs     = -1,        # alle CPUs nutzen
    verbose    = 1
    )



    # Hyperparameter tuning
    grid.fit(X_train, y_train)
    # Resultate zurückgeben
    print("Best Parameters: ", grid.best_params_)
    #print("Best Mean AUC:   ", grid.best_score_)
    print("Best Mean Recall:   ", grid.best_score_)
    
    # Pick best Model 
    best_model = grid.best_estimator_


    """Fit the Model, do Prediction and Probability and Save the Metrics for each Loop"""
    #model = LogisticRegression(class_weight='balanced', max_iter=1000)
    best_model.fit(X_train, y_train)

    # Vorhersagen und Wahrscheinlichkeiten
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]  #für die ROC Kurve

    # Metriken berechnen
    precision_list.append(precision_score(y_test, y_pred))
    recall_list.append(recall_score(y_test, y_pred))
    specificity_list.append(recall_score(y_test, y_pred, pos_label=0))
    f1_list.append(f1_score(y_test, y_pred))
    accuracy_list.append(accuracy_score(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    auc_list.append(auc)

    # Confusion Matrix berechnen
    cm = confusion_matrix(y_test, y_pred)

    conf_matrices.append(cm)

    # ROC-Kurve für jeden Fold
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tprs.append(tpr_interp)

    print(f"Fold {fold}: Precision={precision_list[-1]:.2f}, Recall={recall_list[-1]:.2f}, Specificity={specificity_list[-1]:.2f}, Accuracy={accuracy_list[-1]:.2f} F1={f1_list[-1]:.2f}, AUC={auc:.2f}")


    """Prepare and Save Values for Feature Importance using Shap"""
    # SHAP Explainer für lineare Modelle
    explainer = shap.LinearExplainer(best_model, X_train, feature_perturbation="interventional")

    # SHAP-Werte berechnen und Speichern für jeden Fold
    shap_values = explainer.shap_values(X_test)
    all_shap_vals.append(shap_values)
    all_X_tests.append(X_test)

    #print(shap_values)


    """Feature Importance"""
    # Feature-Importances speichern (absolute Koeffizienten)
    coefs =np.abs(best_model.coef_[0])
    coefs_m.append(coefs)




"""ROC KURVEN FÜR ALLE FOLDS + MENA ROC KURVE"""
# ROC-Kurven aller Folds plotten
plt.figure(figsize=(10, 8))
for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, lw=1, alpha=0.7, label=f"ROC Fold {i + 1} (AUC {i+1} = {auc_list[i]:.2f})")

mean_tpr = np.mean(tprs, axis=0)
std_tpr = np.std(tprs, axis=0)
mean_auc = np.mean(auc_list)
plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC(AUC Mean = {mean_auc:.2f})", lw=1.5)                            # Die Mean ROC Kurve
#plt.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='grey', alpha=0.2, label='± 1 Std. Dev.')      #Der Std Bereich (Grau)
plt.plot([0, 1], [0, 1], linestyle='--', color='red', lw=2, label='Random Classifier')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for all Folds')
plt.legend(loc="lower right")
plt.grid()

# Show Plot comment the line if you want to save it
#plt.show()

#save fig
plt.savefig('../../output/Log_R_output/ROC_kurven_LogR.png')


# Nur Mean Confusion Matrix
"""
# Nur Durchschnittliche ROC-Kurve zeichnen
mean_tpr = np.mean(tprs, axis=0)
mean_auc = np.mean(auc_list)

plt.figure(figsize=(8, 6))
plt.plot(mean_fpr, mean_tpr, label=f'Mean ROC (AUC = {mean_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--',color='r', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stratified K-Fold ROC Curve')
plt.legend(loc="lower right")
plt.grid()
plt.show()
"""


"""FEATURE IMPORTANCE"""
imp_arr  = np.vstack(coefs_m)            # shape (n_folds, n_features)
imp_mean = imp_arr.mean(axis=0)              # mean importance per feature
imp_std  = imp_arr.std(axis=0)               # std dev per featur


df_imp = pd.DataFrame({
    "feature":   X.columns,
    "importance": imp_mean,
    "std":        imp_std
}).sort_values("importance", ascending=False)


# Barplot mit Fehlerbalken
plt.figure(figsize=(8, 6))
sns.barplot(
    data=df_imp,
    x="importance",
    y="feature",
    color="Skyblue",
    xerr=df_imp["std"],   
    capsize=0.2
)
plt.xlabel("Mean |Coefficient| ± 1 StdDev")
plt.title("Feature Importance der LogReg\nüber 5 Outer-Folds")
plt.tight_layout()

# Show Plot comment the line if you want to save it
#plt.show()

#save fig
plt.savefig('../../output/Log_R_output/feature_importance_LogR.png')


"""Shap Feature Evaluation"""
# SHAP-Werte und Korrespondenz-DF zusammenführen
shap_matrix = np.vstack(all_shap_vals)
X_concat    = pd.concat(all_X_tests, axis=0)



# Style setzen (optional, für grauen Hintergrund)
plt.style.use('seaborn-v0_8-whitegrid')


# Beeswarm-Plot
plt.figure(figsize=(10,8))
shap.summary_plot(
    shap_matrix,           # SHAP-Werte für positive Klasse
    X_concat,              # DataFrame mit Feature-Spalten
    plot_type="dot",       # 'dot' ist Default – kann weggelassen werden
    color_bar_label='Feature Value',
    title='SHAP Summary (Beeswarm plot) LogR',
    show=False             # Disable Auto Show
)



# Titel setzen
plt.title("Feature Importance using SHAP (Beeswarm Plot - LogR)",fontsize=14, fontweight="bold", pad=20)
# Nach dem summary_plot:
plt.xlabel("SHAP Impact auf Diabetes-Vorhersage", fontsize=12)

plt.tight_layout()  # Ränder anpassen
plt.savefig('../../output/Log_R_output/Shap_feature_importance_LogR.png')


"""MEAN CONFUSION MATRIX"""
# Mean Confusion Matrix berechnen und plotten
mean_conf_matrix = np.mean(conf_matrices, axis=0) # absolute Mean Werte 

mean_conf_matrix = mean_conf_matrix / mean_conf_matrix.sum(axis=1, keepdims=True) * 100  # Macht aus absoluten relative Werte für Confusion Matrix

plt.figure(figsize=(6, 5))
sns.heatmap(mean_conf_matrix, annot=True, fmt=".1f", cmap="Blues", 
            xticklabels=["No Diabetes ", "Diabetes "],
            yticklabels=["No Diabetes ", "Diabetes "])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Mean Confusion Matrix (%; LogR)')
#plt.show()

# Save fig 
plt.savefig('../../output/Log_R_output/Mean_cm_LogR.png')



"""Zusammenfassungen der Metriken"""
# Zusammenfassung der Metriken
metrics = {
    "F1-Score": (np.mean(f1_list), np.std(f1_list)),
    "Recall": (np.mean(recall_list), np.std(recall_list)),
    #"Specificity": (np.mean(specificity_list), np.std(specificity_list)),
    "Precision": (np.mean(precision_list), np.std(precision_list)),
    "Accuracy": (np.mean(accuracy_list), np.std(accuracy_list)),
    "AUC": (np.mean(auc_list), np.std(auc_list))
}

print("=== Durchschnitt und Standardabweichung der Metriken ===")
for metric, (mean, std) in metrics.items():
    print(f"{metric}: Mittelwert = {mean:.3f}, Standardabweichung = {std:.3f}")
    print(f"{metric}: {mean:.3f} ± {std:.3f} ")
    print("")








# In DataFrame umwandeln
df = pd.DataFrame(metrics).T.reset_index()
df.columns = ['Metric', 'Mean', 'Std']

print(df)


plt.figure(figsize=(8, 5))
ax = sns.barplot(data=df, x='Metric', y='Mean', palette='Blues_d')

# Manuell Fehlerbalken hinzufügen
for i, row in df.iterrows():
    ax.errorbar(
        i,             # x-Position des Balkens
        row['Mean'],   # Mittelwert
        yerr=row['Std'],  # Standardabweichung
        fmt='none',    # kein Marker
        c='black',     # Farbe
        capsize=5      # Querstrichgröße
    )

plt.ylim(0, 1)
plt.ylabel('Mean ± 1 StdDev')
plt.title('Durchschnittliche Metriken mit Fehlerbalken')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# %%
