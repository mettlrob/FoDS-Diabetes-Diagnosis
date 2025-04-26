# Importieren der notwendigen Bibliotheken
import pandas as pd
import numpy as np # Wird für die Behandlung von Nullwerten benötigt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt # Zum Plotten der ROC-Kurve (optional)

# --- 1. Daten laden ---
print("Schritt 1: Daten laden...")
data = pd.read_csv('Dataset/diabetes.csv')

# --- 2. Datenvorverarbeitung ---
print("\nSchritt 2: Datenvorverarbeitung...")
# Spalten, bei denen der Wert 0 wahrscheinlich fehlende Daten darstellt
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Ersetzen von 0 durch NaN (Not a Number) in diesen Spalten
# Dies ist wichtig, da ein Blutzucker von 0 usw. biologisch nicht plausibel ist.
data[cols_with_zeros] = data[cols_with_zeros].replace(0, np.nan)
print(f"\nAnzahl fehlender Werte (NaN) nach Ersetzen von 0 in {cols_with_zeros}:")
print(data[cols_with_zeros].isnull().sum())

# Ersetzen der NaN-Werte durch den Median der jeweiligen Spalte
# Der Median ist oft robuster gegenüber Ausreißern als der Mittelwert.
for col in cols_with_zeros:
    median_val = data[col].median()
    data[col].fillna(median_val, inplace=True)
    print(f"NaN-Werte in Spalte '{col}' mit Median ({median_val:.2f}) aufgefüllt.")

print("\nÜberprüfung auf verbleibende NaN-Werte:")
print(data.isnull().sum().sum()) # Sollte 0 sein

# Trennen in Features (Merkmale) und Target (Zielvariable)
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data[feature_cols] # Features (Eingabevariablen)
y = data['Outcome']    # Target (Ausgabevariable: 0 oder 1)

print("\nFeatures (X) und Target (y) wurden getrennt.")

# --- 3. Daten aufteilen ---
print("\nSchritt 3: Daten aufteilen in Trainings- und Testsets...")
# 80% der Daten für Training, 20% für Testen
# random_state sorgt dafür, dass die Aufteilung bei jeder Ausführung gleich ist (Reproduzierbarkeit)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y sorgt dafür, dass das Verhältnis von 0 und 1 in Trainings- und Testset ähnlich ist wie im Gesamtdatensatz
print(f"Trainingsset Grösse: {X_train.shape[0]} Zeilen")
print(f"Testset Grösse: {X_test.shape[0]} Zeilen")

# --- 4. Random Forest Modell trainieren ---
print("\nSchritt 4: Random Forest Modell trainieren...")
# Erstellen einer Instanz des RandomForestClassifier
# n_estimators: Anzahl der Bäume im Wald (kann angepasst werden)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
# class_weight='balanced' berücksichtigt ungleiche Verteilungen der Klassen (falls vorhanden)

# Trainieren des Modells mit den Trainingsdaten
rf_model.fit(X_train, y_train)
print("Modell erfolgreich trainiert.")

# --- 5. Vorhersagen treffen ---
print("\nSchritt 5: Vorhersagen mit dem Testset treffen...")
# Vorhersagen für das Testset
y_pred = rf_model.predict(X_test)
# Vorhersage der Wahrscheinlichkeiten (für ROC-Kurve benötigt)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
print("Vorhersagen abgeschlossen.")

# --- 6. Modell bewerten ---
print("\nSchritt 6: Modellleistung bewerten...")
# Genauigkeit (Accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Genauigkeit (Accuracy): {accuracy:.4f}")

# Konfusionsmatrix
print("\nKonfusionsmatrix:")
# Zeigt True Positives, False Positives, False Negatives, True Negatives
print(confusion_matrix(y_test, y_pred))

# Klassifikationsbericht
print("\nKlassifikationsbericht:")
# Zeigt Precision, Recall, F1-Score pro Klasse
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC AUC Score: {roc_auc:.4f}") # Fläche unter der ROC-Kurve

# --- 7. Merkmalswichtigkeit (Feature Importance) ---
print("\nSchritt 7: Merkmalswichtigkeit analysieren...")
# Extrahieren der Wichtigkeit jedes Merkmals
importances = rf_model.feature_importances_
features = X.columns

# Erstellen eines DataFrames zur besseren Darstellung
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
# Sortieren nach Wichtigkeit (absteigend)
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nWichtigkeit der Merkmale:")
print(feature_importance_df)

# --- (Optional) ROC-Kurve plotten ---
print("\nSchritt 8: ROC-Kurve plotten (optional)...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonale Linie (Zufallsklassifikator)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Random Forest')
plt.legend(loc="lower right")
# plt.show() # Kommentar entfernen, um den Plot anzuzeigen
print("Plot-Code ausgeführt. Entfernen Sie das Kommentarzeichen vor 'plt.show()', um den Plot anzuzeigen.")
plt.show()