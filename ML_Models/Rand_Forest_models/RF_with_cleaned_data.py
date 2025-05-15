# Importieren der notwendigen Bibliotheken
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score # StratifiedKFold and cross_val_score hinzugefügt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
# --- Schritt 1: Daten laden ---
print("Schritt 1: Daten laden...")
try:
    data = pd.read_csv('Data_Processing/no_transformer_data.csv')
    print("Daten erfolgreich geladen.")
    # print(data.head()) # Optional: Zum Überprüfen der Spalten
except FileNotFoundError:
    print("Fehler: 'data/cleaned_diabetes.csv' wurde nicht gefunden. Bitte überprüfen Sie den Pfad und Dateinamen.") # Korrigierte Fehlermeldung
    exit()

# --- Schritt 2: Features (X) und Zielvariable (y) definieren ---
print("\nSchritt 2: Features (X) und Zielvariable (y) definieren...")
target_column = 'Outcome' # Definieren Sie hier Ihren Zielspaltennamen
if target_column in data.columns:
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    print(f"Features (X) und Zielvariable (y) erfolgreich definiert. Form X: {X.shape}, Form y: {y.shape}")
else:
    print(f"Fehler: Die Zielspalte '{target_column}' wurde nicht in den Daten gefunden.")
    print(f"Verfügbare Spalten: {data.columns.tolist()}")
    exit()

# --- Schritt 3: Daten aufteilen (Trainings- und finaler Testset) ---
# Dieser Testset (X_test, y_test) wird NICHT in der Nested Cross-Validation verwendet,
# sondern dient der finalen Bewertung des Modells, das auf dem gesamten X_train trainiert wurde.
print("\nSchritt 3: Daten in Trainings- und finalen Testset aufteilen...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Trainingsset Größe: {X_train.shape[0]}, Testset Größe: {X_test.shape[0]}")

# --- Schritt 4: Nested Cross-Validation zur robusten Leistungsbewertung ---
print("\nSchritt 4: Nested Cross-Validation starten...")

# Basis-Modell (wird innerhalb von GridSearchCV verwendet)
rf_base = RandomForestClassifier(random_state=42, class_weight='balanced')

# Parametergitter für die innere Schleife (GridSearchCV)
param_grid = {
    'n_estimators': [100, 200], # Reduziert für schnelleres Beispiel, anpassen!
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5], # Reduziert, anpassen!
    'min_samples_leaf': [1, 3]   # Reduziert, anpassen!
}

# Innere Schleife: GridSearchCV für Hyperparameter-Tuning
# verbose=1 oder 0 hier, da es sonst sehr viel Output in der äußeren Schleife gibt
inner_cv_grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                                    cv=3, # Anzahl der Folds für die innere CV (Hyperparameter-Tuning) - Beispiel: 3
                                    n_jobs=-1, scoring='f1', verbose=0) # verbose=0 um Output zu reduzieren

# Äußere Schleife: Cross-Validation zur Bewertung des GridSearchCV-Prozesses
# Wir verwenden StratifiedKFold, da es sich um eine Klassifikationsaufgabe handelt und wir stratify=y in train_test_split verwendet haben.
outer_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Beispiel: 5 äußere Folds

# Führe die Nested Cross-Validation durch.
# cross_val_score führt GridSearchCV (inner_cv_grid_search) für jeden Fold der äußeren CV aus.
# X_train und y_train werden hier verwendet, um die Leistung auf dem Trainingsdatensatz robust zu schätzen.
print(f"Nested CV wird auf X_train ({X_train.shape[0]} Samples) durchgeführt...")
nested_scores = cross_val_score(inner_cv_grid_search, X=X_train, y=y_train, cv=outer_cv_folds, scoring='roc_auc', n_jobs=-1)

print(f"\nNested ROC AUC Scores für jeden äußeren Fold: {nested_scores}")
print(f"Durchschnittlicher Nested ROC AUC Score: {nested_scores.mean():.4f}")
print(f"Standardabweichung des Nested ROC AUC Scores: {nested_scores.std():.4f}")
print("Nested Cross-Validation abgeschlossen.")

# --- Schritt 5: Training des finalen Modells mit GridSearchCV auf dem gesamten Trainingsset ---
# Nachdem wir eine robuste Schätzung der Leistung durch Nested CV erhalten haben,
# trainieren wir nun das Modell mit den besten Hyperparametern auf dem *gesamten* X_train Datensatz,
# um es anschließend auf dem X_test Datensatz zu bewerten.
print("\nSchritt 5: Training des finalen Modells mit GridSearchCV auf dem gesamten X_train...")

# Wir verwenden hier eine neue GridSearchCV-Instanz oder fitten die alte neu,
# um die besten Parameter für das *gesamte* X_train zu finden.
# verbose=2 hier, um den Prozess zu sehen. cv=5 wie im Originalcode.
final_grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid,
                                 cv=5, n_jobs=-1, scoring='roc_auc', verbose=2)

final_grid_search.fit(X_train, y_train)

print("\nBeste gefundene Parameter für das finale Modell (auf X_train):")
print(final_grid_search.best_params_)

best_rf_model = final_grid_search.best_estimator_
print("\nBestes finales Modell wurde ausgewählt und auf X_train trainiert.")

# --- Schritt 6: Vorhersagen mit dem finalen OPTIMIERTEN Modell auf X_test ---
print("\nSchritt 6: Vorhersagen mit dem finalen optimierten Modell auf X_test treffen...")
y_pred = best_rf_model.predict(X_test)
y_pred_proba = best_rf_model.predict_proba(X_test)[:, 1]
print("Vorhersagen abgeschlossen.")

# --- Schritt 7: Leistung des finalen OPTIMIERTEN Modells auf X_test bewerten ---
print("\nSchritt 7: Leistung des finalen optimierten Modells auf X_test bewerten...")
accuracy = accuracy_score(y_test, y_pred)
print(f"Finale Genauigkeit (Accuracy) auf X_test: {accuracy:.4f}") # Angepasste Bezeichnung
print("\nFinale Konfusionsmatrix auf X_test:") # Angepasste Bezeichnung
print(confusion_matrix(y_test, y_pred))
print("\nFinaler Klassifikationsbericht auf X_test:") # Angepasste Bezeichnung
print(classification_report(y_test, y_pred))
roc_auc_test = roc_auc_score(y_test, y_pred_proba) # Eigene Variable für Klarheit
print(f"\nFinaler ROC AUC Score auf X_test: {roc_auc_test:.4f}") # Angepasste Bezeichnung

cm = confusion_matrix(y_test, y_pred)
print("\nPlotting der Konfusionsmatrix auf X_test...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Konfusionsmatrix - Finales Modell auf X_test")
plt.savefig("output/RFC_output/confusion_matrix.png") 
plt.show()


# --- Schritt 8: Merkmalswichtigkeit des finalen OPTIMIERTEN Modells ---
print("\nSchritt 8: Merkmalswichtigkeit des finalen Modells analysieren...")
importances = best_rf_model.feature_importances_
# X.columns ist hier in Ordnung, da X_train die gleichen Spalten wie X hat
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print("\nWichtigkeit der Merkmale (finales optimiertes Modell):")
print(feature_importance_df)


print("\nPlotting der Merkmalswichtigkeit...")
plt.figure(figsize=(10, 8)) # Eventuell Größe anpassen, je nach Anzahl Features
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, color='blue')
plt.title('Merkmalswichtigkeit - Finales Random Forest Modell')
plt.xlabel('Wichtigkeit (Importance)')
plt.ylabel('Merkmal (Feature)')
plt.tight_layout() # Passt den Plot an, um Überlappungen zu vermeiden
plt.savefig("output/RFC_output/feature_importance_plot.png")
plt.show()



# --- Schritt 9: ROC-Kurve des finalen OPTIMIERTEN Modells auf X_test plotten ---
print("\nSchritt 9: ROC-Kurve des finalen Modells auf X_test plotten...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Finale ROC curve auf X_test (area = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Finale ROC Curve - Random Forest (auf X_test)')
plt.legend(loc="lower right")
# plt.show() # Kommentar entfernen, um den Plot anzuzeigen
print("Plot-Code ausgeführt. Entfernen Sie ggf. das Kommentarzeichen vor 'plt.show()', um den Plot anzuzeigen.")
plt.savefig("output/RFC_output/roc_curve.png")
plt.show()
