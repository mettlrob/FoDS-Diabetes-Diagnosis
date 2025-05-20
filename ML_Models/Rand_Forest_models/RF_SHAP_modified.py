# Importieren der notwendigen Bibliotheken
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             roc_auc_score, roc_curve, f1_score, recall_score, precision_score)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import os # Für das Erstellen von Ordnern
import shap

# --- Schritt 1: Daten laden ---
print("Schritt 1: Daten laden...")
try:
    # Stelle sicher, dass der Pfad zu deiner Datei korrekt ist
    data = pd.read_csv('Data_Processing/no_transformer_data.csv')
    print("Daten erfolgreich geladen.")
except FileNotFoundError:
    print("Fehler: 'Data_Processing/no_transformer_data.csv' wurde nicht gefunden. Bitte überprüfen Sie den Pfad und Dateinamen.")
    exit()
except Exception as e:
    print(f"Ein Fehler beim Laden der Daten ist aufgetreten: {e}")
    exit()

# --- Schritt 2: Features (X) und Zielvariable (y) definieren ---
print("\nSchritt 2: Features (X) und Zielvariable (y) definieren...")
target_column = 'Outcome'
if target_column in data.columns:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print(f"Features (X) und Zielvariable (y) erfolgreich definiert. Form X: {X.shape}, Form y: {y.shape}")
else:
    print(f"Fehler: Die Zielspalte '{target_column}' wurde nicht in den Daten gefunden.")
    print(f"Verfügbare Spalten: {data.columns.tolist()}")
    exit()

# --- Schritt 3: Daten aufteilen (Trainings- und finaler Testset) ---
print("\nSchritt 3: Daten in Trainings- und finalen Testset aufteilen...")
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Trainingsset Größe: {X_train.shape[0]}, Testset Größe: {X_test.shape[0]}")
except Exception as e:
    print(f"Fehler beim Aufteilen der Daten: {e}")
    exit()


# --- MODIFIZIERTER Schritt 4: Nested Cross-Validation mit ROC-Kurven-Sammlung ---
print("\nSchritt 4: Nested Cross-Validation starten (manuelle Schleife für ROC-Daten)...")


param_grid = {
    'n_estimators': [5, 10, 50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 3]
}
innercv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
outer_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



all_fprs = [] # NEU: Liste für die FPRs jedes Folds
all_tprs_raw = [] # NEU: Liste für die rohen TPRs jedes Folds (nicht interpoliert)
tprs_interp_list = [] # Umbenannt von tprs_list für Klarheit (interpolierte TPRs)
aucs_list = []
f1_scores_list = []
recall_scores_list = []
precision_scores_list = []
accuracy_scores_list = []
mean_fpr = np.linspace(0, 1, 100)

#initializing for shap
shap_models = []
shap_test_sets = []

main_output_folder = "output/RFC_output"
# output_folder_path_cv = os.path.join(main_output_folder, "cv_metrics") # Nicht mehr zwingend nötig für CSVs

os.makedirs(main_output_folder, exist_ok=True) # Erstelle den Hauptausgabeordner, falls nicht vorhanden

X_train_np = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
y_train_np = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

print(f"Nested CV wird auf X_train ({X_train_np.shape[0]} Samples) durchgeführt...")
for i, (train_idx, val_idx) in enumerate(outer_cv_folds.split(X_train_np, y_train_np)):
    print(f"  Äußerer Fold {i+1}/{outer_cv_folds.get_n_splits()}...")

    #Split into outer train/validation 
    X_outer_train, X_outer_val = X_train_np[train_idx], X_train_np[val_idx]
    y_outer_train, y_outer_val = y_train_np[train_idx], y_train_np[val_idx]

    rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

    inner_cv_grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                                    cv= innercv,
                                    n_jobs=-1, scoring='roc_auc', verbose=0)

    inner_cv_grid_search.fit(X_outer_train, y_outer_train)
    best_model_fold = inner_cv_grid_search.best_estimator_
    
#    Speichern des besten Modells für SHAP-Analyse
    shap_models.append(best_model_fold)
    shap_test_sets.append(pd.DataFrame(X_outer_val, columns = X.columns))

    y_pred_proba_fold = best_model_fold.predict_proba(X_outer_val)[:, 1]
    y_pred_class_fold = best_model_fold.predict(X_outer_val)
    
    fpr_fold, tpr_fold, _ = roc_curve(y_outer_val, y_pred_proba_fold)
    roc_auc_fold = roc_auc_score(y_outer_val, y_pred_proba_fold)

    all_fprs.append(fpr_fold) # NEU: Speichere FPR dieses Folds
    all_tprs_raw.append(tpr_fold) # NEU: Speichere TPR dieses Folds
    aucs_list.append(roc_auc_fold)
    
    f1_fold = f1_score(y_outer_val, y_pred_class_fold, pos_label=1, zero_division=0)
    f1_scores_list.append(f1_fold)
    
    recall_fold = recall_score(y_outer_val, y_pred_class_fold, pos_label=1, zero_division=0)
    recall_scores_list.append(recall_fold)

    precision_fold = precision_score(y_outer_val, y_pred_class_fold, pos_label=1, zero_division=0)
    precision_scores_list.append(precision_fold)
    
    accuracy_fold = accuracy_score(y_outer_val, y_pred_class_fold)
    accuracy_scores_list.append(accuracy_fold)
    
    tprs_interp_list.append(np.interp(mean_fpr, fpr_fold, tpr_fold))
    tprs_interp_list[-1][0] = 0.0
    print(f"    Fold {i+1} Metriken: AUC={roc_auc_fold:.4f}, F1={f1_fold:.4f}, Recall={recall_fold:.4f}, Precision={precision_fold:.4f}, Acc={accuracy_fold:.4f}")

print("\nNested Cross-Validation (manuelle Schleife) abgeschlossen.")

""" --- SHAP-Analysis and Plotting ---"""
shap_values_all = []
all_X_test = []

for model, X_test_fold in zip(shap_models, shap_test_sets):
    explainer = shap.TreeExplainer(model)
    shap_values_fold = explainer.shap_values(X_test_fold)

    #For binary classification, use SHAP values for positive class
    if isinstance(shap_values_fold, list):
        shap_values_positive = shap_values_fold[1]
    elif shap_values_fold.ndim == 3:
        shap_values_positive = shap_values_fold[:, :, 1] #take class 1

    else:
        shap_values_positive = shap_values_fold

    
    print(f"Fold SHAP shape: {shap_values_positive.shape}, X_test shape: {X_test_fold.shape}")
    shap_values_all.append(shap_values_positive)
    all_X_test.append(X_test_fold)


#Stack values
shap_values_concat = np.vstack(shap_values_all)
X_test_concat = pd.concat(all_X_test, axis = 0, ignore_index= True)
X_test_concat = pd.DataFrame(X_test_concat, columns = X.columns)

print("SHAP values shape:", shap_values_concat.shape)
print("X_test_concat shape:", X_test_concat.shape)


plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10,8))
shap.summary_plot(shap_values_concat,
                  X_test_concat,
                  plot_type = 'dot',
                  color_bar_label = 'Feature Value',
                  show = False)
                   
                   
plt.title("Random Forest - SHAP Feature Importance\n(Aggregated Across Outer Folds)", fontsize=14, fontweight='bold', pad=20)
plt.xlabel("SHAP Value (Impact on Predicting Diabetes)", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.tight_layout()
plt.savefig("output/RFC_output/rfc_shap_summary_plot.png", dpi=300)
plt.close()


print(f"\nDurchschnittlicher Nested ROC AUC Score: {np.mean(aucs_list):.4f} (Std: {np.std(aucs_list):.4f})")
print(f"Durchschnittlicher Nested F1 Score (pos_label=1): {np.mean(f1_scores_list):.4f} (Std: {np.std(f1_scores_list):.4f})")
print(f"Durchschnittlicher Nested Recall Score (pos_label=1): {np.mean(recall_scores_list):.4f} (Std: {np.std(recall_scores_list):.4f})")
print(f"Durchschnittlicher Nested Accuracy Score: {np.mean(accuracy_scores_list):.4f} (Std: {np.std(accuracy_scores_list):.4f})")



# --- ANGEPASSTER Schritt 4.1: Speichern der CV-Metriken-Zusammenfassung im gewünschten Format ---
print("\nSchritt 4.1: Speichern der Cross-Validation Metriken Zusammenfassung...")

# Erstelle das Dictionary im gewünschten Format
cv_summary_data = {
    'Model': ['RandomForest_CV'], # Modellname für die CV-Ergebnisse
    'F1_Mean': [np.mean(f1_scores_list)],
    'F1_Std': [np.std(f1_scores_list)],
    'Recall_Mean': [np.mean(recall_scores_list)],
    'Recall_Std': [np.std(recall_scores_list)],
    'Precision_Mean': [np.mean(precision_scores_list)], # NEU
    'Precision_Std': [np.std(precision_scores_list)],   # NEU
    'Accuracy_Mean': [np.mean(accuracy_scores_list)],
    'Accuracy_Std': [np.std(accuracy_scores_list)],
    'ROC_AUC_Mean': [np.mean(aucs_list)],
    'ROC_AUC_Std': [np.std(aucs_list)]
}

cv_summary_df = pd.DataFrame(cv_summary_data)

# Dateipfad für die Zusammenfassungs-CSV im Data_Processing Ordner
# Der Dateiname im Screenshot deines Kollegen ist 'rfc_summary_metrics.csv'
# Du könntest einen ähnlichen Namen wählen oder ihn anpassen
summary_filename = "rfc_cv_summary_metrics.csv" # Angepasster Name für Klarheit (CV Summary)
cv_summary_filepath = os.path.join("Data_Processing", summary_filename)

cv_summary_df.to_csv(cv_summary_filepath, index=False)
print(f"Zusammenfassung der CV-Metriken gespeichert in: {cv_summary_filepath}")

# Die CSV-Datei mit den Metriken pro Fold wird nicht mehr erstellt, wie gewünscht.
# Wenn du sie doch brauchst, kannst du den alten Code für cv_metrics_per_fold_df wieder aktivieren.


# --- Schritt 4.5: Mittlere ROC-Kurve plotten und finalisieren ---
# --- KORRIGIERTER Schritt 4.5: Plotten ALLER Fold-ROC-Kurven, Mittlere ROC-Kurve und Standardabweichung ---
print("\nSchritt 4.5: Plotten der ROC-Kurven (alle Folds und Mittelwert)...")
plt.figure(figsize=(10, 8))

# Plotten jeder einzelnen ROC-Kurve pro Fold
# all_fprs, all_tprs_raw und aucs_list wurden in Schritt 4 gefüllt
for i in range(len(all_fprs)):
    plt.plot(all_fprs[i], all_tprs_raw[i], lw=1, alpha=0.4, # Dünnere, halbtransparente Linien
             label=f'ROC Fold {i+1} (AUC = {aucs_list[i]:.2f})' if i < 5 else None) # Legende nur für die ersten paar Folds, sonst unübersichtlich

# Chance-Linie
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='grey', label='Chance', alpha=.8)
"""
# Mittlere TPR und AUC (berechnet mit interpolierten TPRs aus tprs_interp_list)
mean_tpr = np.mean(tprs_interp_list, axis=0) # tprs_interp_list wurde in Schritt 4 korrekt gefüllt
mean_tpr[-1] = 1.0
mean_auc_val = np.mean(aucs_list)
std_auc_val = np.std(aucs_list)
plt.plot(mean_fpr, mean_tpr, color='blue', # mean_fpr wurde in Schritt 4 definiert
         label=f'Mean ROC (AUC = {mean_auc_val:.2f} $\\pm$ {std_auc_val:.2f})',
         lw=2.5, alpha=.9) # Dickere Linie für Mittelwert

# Standardabweichungsband
std_tpr = np.std(tprs_interp_list, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='cornflowerblue', alpha=.3,
                 label=r'$\pm$ 1 std. dev.')
"""
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves from Nested CV - Random Forest (on X_train)')
plt.legend(loc="lower right", fontsize='small')
plt.grid(alpha=0.3)
nested_cv_roc_plot_path = os.path.join(main_output_folder, "nested_cv_roc_curves_with_folds_RFC.png")
plt.savefig(nested_cv_roc_plot_path)
print(f"Nested CV ROC Kurve (mit Folds) gespeichert in: {nested_cv_roc_plot_path}")


# --- Schritt 5: Training des finalen Modells mit GridSearchCV auf dem gesamten Trainingsset ---
print("\nSchritt 5: Training des finalen Modells mit GridSearchCV auf dem gesamten X_train...")
final_grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                                 cv=5,
                                 n_jobs=-1, scoring='roc_auc', verbose=1)
final_grid_search.fit(X_train, y_train)
print("\nBeste gefundene Parameter für das finale Modell (auf X_train):")
print(final_grid_search.best_params_)
best_rf_model = final_grid_search.best_estimator_
print("\nBestes finales Modell wurde ausgewählt und auf X_train trainiert.")

# --- Schritt 5.5: Visualisierung eines einzelnen Entscheidungsbaums (optional) ---
if hasattr(best_rf_model, 'estimators_') and len(best_rf_model.estimators_) > 0:
    print("\nSchritt 5.5: Visualisierung eines einzelnen Entscheidungsbaums...")
    try:
        single_tree = best_rf_model.estimators_[0]
        plt.figure(figsize=(20,10))
        plot_tree(single_tree,
                  feature_names=X.columns.tolist(),
                  class_names=[str(cls) for cls in best_rf_model.classes_],
                  filled=True, rounded=True, impurity=True, proportion=False,
                  fontsize=7, max_depth=3)
        plt.title("Visualisierung eines einzelnen Entscheidungsbaums aus dem Random Forest (max_depth=3)")
        tree_plot_path = os.path.join(main_output_folder, "single_decision_tree_from_rf.png")
        plt.savefig(tree_plot_path)
        print(f"Einzelner Entscheidungsbaum visualisiert und gespeichert in: {tree_plot_path}")
        
    except Exception as e:
        print(f"Fehler bei der Visualisierung des Baumes: {e}")
else:
    print("\nSchritt 5.5: Konnte keinen einzelnen Baum visualisieren (Modell hat keine 'estimators_').")


# --- Schritt 6: Vorhersagen mit dem finalen OPTIMIERTEN Modell auf X_test ---
print("\nSchritt 6: Vorhersagen mit dem finalen optimierten Modell auf X_test treffen...")
y_pred_final = best_rf_model.predict(X_test)
y_pred_proba_final = best_rf_model.predict_proba(X_test)[:, 1]
print("Vorhersagen auf X_test abgeschlossen.")

# --- Schritt 7: Leistung des finalen OPTIMIERTEN Modells auf X_test bewerten ---
# HIER WIRD DIE CSV FÜR DIE FINALEN METRIKEN ERSTELLT (WIE ZUVOR BESPROCHEN)
print("\nSchritt 7: Leistung des finalen optimierten Modells auf X_test bewerten...")
accuracy_final = accuracy_score(y_test, y_pred_final) # y_pred_final aus Schritt 6
roc_auc_final_test = roc_auc_score(y_test, y_pred_proba_final) # y_pred_proba_final aus Schritt 6
print(f"Finale Genauigkeit (Accuracy) auf X_test: {accuracy_final:.4f}")

print("\nFinale Konfusionsmatrix auf X_test:")
cm_final = confusion_matrix(y_test, y_pred_final)
print(cm_final)

print("\nFinaler Klassifikationsbericht auf X_test:")
report_final_dict = classification_report(y_test, y_pred_final, zero_division=0, output_dict=True)
print(classification_report(y_test, y_pred_final, zero_division=0))
print(f"\nFinaler ROC AUC Score auf X_test: {roc_auc_final_test:.4f}")

# --- Speichern der finalen Test-Metriken in einer CSV-Datei (im Data_Processing Ordner) ---
print("\nSpeichern der finalen Test-Metriken in CSV im Ordner Data_Processing...")
final_test_metrics_summary_data = {
    'Model': ['RandomForest_FinalTest'], # Unterscheidungsmerkmal
    'F1_Mean_Class1': [report_final_dict['1']['f1-score']], # F1-Score für Klasse 1
    'F1_Std_Class1': [0], # Keine Std für einen einzelnen Testlauf
    'Recall_Mean_Class1': [report_final_dict['1']['recall']], # Recall für Klasse 1
    'Recall_Std_Class1': [0],
    'Precision_Mean_Class1': [report_final_dict['1']['precision']], # Precision für Klasse 1
    'Precision_Std_Class1': [0],
    'Accuracy_Mean': [accuracy_final],
    'Accuracy_Std': [0],
    'ROC_AUC_Mean': [roc_auc_final_test],
    'ROC_AUC_Std': [0]
}
final_test_summary_df = pd.DataFrame(final_test_metrics_summary_data)

# Dateipfad für die finale Test-Metriken-CSV im Data_Processing Ordner
final_test_summary_filename = "rfc_final_test_summary_metrics.csv"
final_test_summary_filepath = os.path.join("Data_Processing", final_test_summary_filename)

final_test_summary_df.to_csv(final_test_summary_filepath, index=False)
print(f"Zusammenfassung der finalen Test-Metriken gespeichert in: {final_test_summary_filepath}")
# --- Ende Speichern der finalen Test-Metriken ---

# Plotten der finalen Konfusionsmatrix (normalisiert zeilenweise)
print("\nPlotting der normalisierten Konfusionsmatrix (Zeilen = 100%) für X_test...")
plt.figure(figsize=(8, 6))
cm_final_normalized_row = confusion_matrix(y_test, y_pred_final, normalize='true') # Normalisierung hier
sns.heatmap(cm_final_normalized_row *100, annot=True, fmt=".2f", cmap="Blues", cbar=True, # Formatierung als Prozent
            xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Random Forest Konfusionsmatrix (in Prozent)") # Aktueller Titel
cm_plot_path = os.path.join(main_output_folder, "RFC_confusion_matrix_Prozent.png")
plt.savefig(cm_plot_path)
print(f"Finale Konfusionsmatrix gespeichert in: {cm_plot_path}")


# --- Schritt 8: Merkmalswichtigkeit des finalen OPTIMIERTEN Modells ---
print("\nSchritt 8: Merkmalswichtigkeit des finalen Modells analysieren...")
if hasattr(best_rf_model, 'feature_importances_'):
    importances = best_rf_model.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print("\nWichtigkeit der Merkmale (finales optimiertes Modell):")
    print(feature_importance_df.to_string())

    print("\nPlotting der Merkmalswichtigkeit...")
    plt.figure(figsize=(10, max(6, len(features)*0.5)))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_r')
    plt.title('Feature Importance Random Forest')
    plt.xlabel('Wichtigkeit (Importance)')
    plt.ylabel('Merkmal (Feature)')
    plt.tight_layout()
    fi_plot_path = os.path.join(main_output_folder, "feature_importance_plot_RFC.png")
    plt.savefig(fi_plot_path)
    print(f"Merkmalswichtigkeits-Plot gespeichert in: {fi_plot_path}")
    
else:
    print("Konnte Merkmalswichtigkeit nicht bestimmen (Modell hat kein 'feature_importances_').")





# --- Schritt 9: ROC-Kurve des finalen OPTIMIERTEN Modells auf X_test plotten ---
print("\nSchritt 9: ROC-Kurve des finalen Modells auf X_test plotten...")
fpr_final_test, tpr_final_test, _ = roc_curve(y_test, y_pred_proba_final)
plt.figure(figsize=(8, 6))
plt.plot(fpr_final_test, tpr_final_test, color='darkorange', lw=2, label=f'Finale ROC curve auf X_test (AUC = {roc_auc_final_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sentessitivity)')
plt.title('Finale ROC Curve - Random Forest (auf X_test)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
final_roc_plot_path = os.path.join(main_output_folder, "roc_curve_final_model_RFC.png")
plt.savefig(final_roc_plot_path)
print(f"Finale ROC-Kurve gespeichert in: {final_roc_plot_path}")


print("\nSkript vollständig ausgeführt.")
    # SHAP-Werte berechnen und Speichern für jeden Fold
