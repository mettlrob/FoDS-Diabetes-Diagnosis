#%%
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# ---- Importing custom module for plotting ----
import os
import sys
# Ensure the project root is on sys.path so we can import support package
this_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, '..'))
sys.path.insert(0, project_root)
from support.plotting_module import plot_roc_curves, plot_confusion_matrices



# ---- Temporary Debug Imputer ----
from sklearn.impute import KNNImputer as SKKNNImputer
class DebugKNNImputer(SKKNNImputer):
    """
    A KNNImputer subclass to log missing values before/after transformation.
    Replace KNNImputer with DebugKNNImputer in the pipeline to verify behavior.
    """
    def fit(self, X, y=None):
        print(f"[DEBUG] Before fit: missing in training set: {np.isnan(X).sum()} values")
        super().fit(X, y)
        return self

    def transform(self, X):
        Xt = super().transform(X)
        print(f"[DEBUG] After transform: missing in data: {np.isnan(Xt).sum()} values")
        return Xt

# --------------------------------



def load_data():
    file_path = '../../data/nan_df.csv' # Adjust the path as needed
    df = pd.read_csv(file_path)
    X = df.drop(columns = ['Outcome'])
    y = df['Outcome']
    return X, y

def evaluate_models(X, y):
    #ideal imputer was defined from proc_data.ipynb in a cross-validation test loop.
    #imputer = KNNImputer(n_neighbors=5)
    imputer = SimpleImputer(strategy='median')  # Use SimpleImputer for simplicity in this example
    scaler = StandardScaler()

    #model selection
    models = {
        'LogisticRegression' : LogisticRegression(solver = 'saga', class_weight = 'balanced', max_iter= 1000, random_state= 42),
        'SVC' : SVC(probability = True, class_weight = 'balanced', random_state= 42),
        'KNN' : KNeighborsClassifier(),
        'RandomForest' : RandomForestClassifier(class_weight = 'balanced', bootstrap= True, random_state= 42)
    }

    # Hyperparameter grids for each model
    param_grids = {
        'LogisticRegression' : {
            #'clf__' prefix refers to the pipeline step named 'clf', so 'C' becomes 'clf__C'.
            'clf__C': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15],   #0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.2, 0.25, 0.3, 1, 10, 100
            'clf__penalty': ['l2', 'l1', None]

        },
        'SVC' : {
            'clf__C': [0.01, 0.1, 1, 10, 100],
            'clf__kernel': ['linear'],
            #'clf__gamma': ['scale', 'auto']
        },
        'KNN' : {
            'clf__n_neighbors': [3, 5, 7, 15, 20],
            'clf__weights': ['uniform', 'distance']
        },
        'RandomForest' : {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 5, 8],
            'clf__min_samples_split': [5, 10, 20],
            'clf__min_samples_leaf': [2, 5, 10],
            'clf__max_features': ['sqrt']
        }
    }

    # Scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'roc_auc': 'roc_auc',
        'f1': 'f1'
    }
    # CV splitters
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize results dictionary (container for all results)

    all_results = {}

    # Loop through each model
    for name, base_clf in models.items():
        print(f"\n### Evaluating {name}... ###")

        #build the pipeline
        pipeline = Pipeline([
            ('imputer', imputer),
            ('scaler', scaler),
            ('clf', base_clf),     
        ], verbose= False)

        # ---- KNN processing steps? ----#



        #------------------------------#

        # Inner-loop GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring=scoring,
            refit='recall',  # Refit on recall for best model selection
            cv=inner_cv,
            n_jobs=-1,
            verbose=1,
            return_train_score=False 
        )

        # Outer-loop cross-validation
        cv_results = cross_validate(
            estimator = grid_search,
            X = X,
            y = y,
            cv=outer_cv,
            scoring=scoring,
            return_train_score=True,
            return_estimator=True,
            n_jobs=1
        )
        # for fold_idx, gs in enumerate(cv_results['estimator']):
        #     print(f"\n— Fold {fold_idx+1} inner‐CV results —")
        #     print(" Best params:", gs.best_params_)
        #     # if you enabled multi‐metric scoring:
        #     print(" F1  at best params:", gs.best_score_)
        #     print(" Prec at best params:", gs.cv_results_['mean_test_precision'][gs.best_index_])
        for fold_idx, gs in enumerate(cv_results['estimator']):
            print(f"\nFold {fold_idx+1} best inner‐CV params: {gs.best_params_}")
            for metric in scoring.keys():  
                key = f"mean_test_{metric}"
             # Some metrics (like roc_auc) may need a different cv_results_ key, but in general:
                print(f"  {metric:8s} = {gs.cv_results_[key][gs.best_index_]:.3f}")

        # Store the raw cross-validate results
        all_results[name] = cv_results 

 

        # Prepare a summary of results
        print(f'\n Summary for {name}:')
        for metric in scoring:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            print(
                f' {metric:10s} train: {np.mean(train_scores):.3f} ± {np.std(train_scores):.3f} '
                f'test: {np.mean(test_scores):.3f} ± {np.std(test_scores):.3f}'
            )
    return all_results

def main():

    #Load data here
    X, y = load_data()

    #Run nested CV evaluation
    results = evaluate_models(X, y)
    
    # #Reconstruct outer CV splits for consistent plotting
    # outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # splits = list(outer_cv.split(X, y))

    #Plot and save
    # plot_roc_curves(results, X.values, y.values, splits)
    # plot_confusion_matrices(results, X.values, y.values, splits)



if __name__ == "__main__":
    main()


