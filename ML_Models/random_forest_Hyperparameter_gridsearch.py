# Importieren der notwendigen Bibliotheken
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV hinzugefügt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# --- Schritt 1: Daten laden ---
print("Schritt 1: Daten laden...")
try:
    data = pd.read_csv('Dataset/diabetes.csv')
    print("Daten erfolgreich geladen.")
   
except FileNotFoundError:
    print("Fehler: 'diabetes.csv' wurde nicht gefunden.")
    exit()

# --- Schritt 2: Datenvorverarbeitung ---
print("\nSchritt 2: Datenvorverarbeitung...")
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
for col in cols_with_zeros:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols]
y = data['Outcome']
print("Vorverarbeitung abgeschlossen.")

# --- Schritt 3: Daten aufteilen ---
print("\nSchritt 3: Daten aufteilen...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Trainingsset: {X_train.shape[0]}, Testset: {X_test.shape[0]}")

# --- Schritt 4: Hyperparameter-Tuning mit GridSearchCV ---
# (Ersetzt das einfache Modelltraining)
print("\nSchritt 4: Hyperparameter-Tuning starten...")

# Definiere das Parametergitter (Beispiel, anpassen!)
param_grid = {
    'n_estimators': [100, 200, 300], # Reduziert für schnelleres Beispiel
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10], # Hinzugefügt
    'min_samples_leaf': [1, 3, 5] ,  # Hinzugefügt
    
}

# Basis-Modell
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# GridSearchCV initialisieren
# Wähle scoring='roc_auc' oder z.B. scoring='f1_weighted' oder scoring='recall_macro'
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                           cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)

# Suche auf Trainingsdaten durchführen
grid_search.fit(X_train, y_train)

# Beste gefundene Parameter
print("\nBeste gefundene Parameter:")
print(grid_search.best_params_)

# Bestes Modell aus der Suche auswählen
best_rf_model = grid_search.best_estimator_
print("\nBestes Modell wurde ausgewählt.")

# --- Schritt 5: Vorhersagen mit dem OPTIMIERTEN Modell ---
print("\nSchritt 5: Vorhersagen mit dem optimierten Modell treffen...")
# Wichtig: Verwende jetzt best_rf_model
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
print("Vorhersagen abgeschlossen.")

# --- Schritt 6: Leistung des OPTIMIERTEN Modells bewerten ---
print("\nSchritt 6: Leistung des optimierten Modells bewerten...")
# Wichtig: Bewerte das best_rf_model
accuracy = accuracy_score(y_test, y_pred)
print(f"Optimierte Genauigkeit (Accuracy): {accuracy:.4f}")
print("\nOptimierte Konfusionsmatrix:")
print(confusion_matrix(y_test, y_pred))
print("\nOptimierter Klassifikationsbericht:")
print(classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nOptimierter ROC AUC Score: {roc_auc:.4f}")

# --- Schritt 7: Merkmalswichtigkeit des OPTIMIERTEN Modells ---
print("\nSchritt 7: Merkmalswichtigkeit analysieren...")
# Wichtig: Verwende best_rf_model
importances = best_rf_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nWichtigkeit der Merkmale (optimiertes Modell):")
print(feature_importance_df)

# --- Schritt 8: ROC-Kurve des OPTIMIERTEN Modells plotten (optional) ---
print("\nSchritt 8: ROC-Kurve plotten (optional)...")
# Wichtig: Verwende y_pred_proba vom best_rf_model
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Optimierte ROC curve (area = {roc_auc:.2f})') # Aktualisierter Label
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Optimierte ROC Curve - Random Forest') # Aktualisierter Titel
plt.legend(loc="lower right")
# plt.show() # Kommentar entfernen, um den Plot anzuzeigen
print("Plot-Code ausgeführt. Entfernen Sie das Kommentarzeichen vor 'plt.show()', um den Plot anzuzeigen.")
plt.show()