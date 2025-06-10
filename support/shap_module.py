import os
import numpy as np
import shap
import matplotlib.pyplot as plt

def plot_shap_summary(results, X, y, splits, outdir = '../pipeline_output/shap/'):
    """
    Generate a SHAP summary plot per model, aggregating across all outer folds. 
    - results: dict from evaluate_models, containing 'estimator' list per model
    - X, y: pandas DataFrame/Series of full dataset
    - splits: list of (train_idx, test_idx) from the same StratifiedKFold
    - outdir: directory to save PNGs

    """
    os.makedirs(outdir, exist_ok = True)

    feature_names = X.columns

    for name, cv_results in results.items():
        
        #accumulate SHAP-values and features
        all_shap_values = []
        all_feature_data = []

        for i, (_, test_idx) in enumerate(splits):
            gs = cv_results['estimator'][i] #gs is GridSearchCV object
            pipeline = gs.best_estimator_
            clf = pipeline.named_steps['clf']
            
            #prepare test data
            X_test = X.iloc[test_idx]
            #apply fitted preprocessing
            X_imp = pipeline.named_steps['imputer'].transform(X_test)
            X_scaled = pipeline.named_steps['scaler'].transform(X_imp)

            model_type = type(clf).__name__.lower()

            # TreeExplainer for RandomForest
            if 'randomforest' in model_type:
                explainer = shap.TreeExplainer(clf)
                shap_vals = explainer.shap_values(X_scaled)

                print("RandomForest classes_: ", clf.classes_)
                print(f"Raw TreeExplainer SHAP shape: {np.shape(shap_vals)}")

                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # shape = (n_samples, n_features, n_classes)
                    if 1 in clf.classes_:
                        class_index = list(clf.classes_).index(1)
                    else:
                        class_index = 0
                    shap_vals = shap_vals[:, :, class_index]
                    print(f"Selected class SHAP shape: {shap_vals.shape}")
                elif isinstance(shap_vals, list):  # fallback for older SHAP versions
                    shap_vals = shap_vals[1]
                else:
                    raise ValueError(f"Unexpected SHAP format for RandomForest: {type(shap_vals)}, shape: {np.shape(shap_vals)}")

            # KernelExplainer for KNeighborsClassifier
            elif 'kneighbors' in model_type or 'knn' in model_type:
                # Use all data points as background
                background = X_scaled #computationally unfeasible for large datasets
                explainer = shap.KernelExplainer(clf.predict_proba, background)
                shap_vals = explainer.shap_values(X_scaled, n_samples = None)

                # ---- fix shape: (n_samples, n_features, n_classes) ----
                
                print("KNN classes_: ", clf.classes_)
                print(f"Raw shap_vals shape: {np.shape(shap_vals)}")

                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                    # Select shap values for the positive class:
                    if 1 in clf.classes_:
                        class_index = list(clf.classes_).index(1)
                    else:
                        class_index = 0
                    shap_vals = shap_vals[:, :, class_index]

                    print(f"Selected class SHAP shape: {shap_vals.shape}")
                else:
                    raise ValueError(
                        f"KNN SHAP values not in expected shape. Got {np.shape(shap_vals)}"
                    )

            # LinearExplainer for LogisticRegression, and SVC using linear kernel
            elif 'logisticregression' in model_type or ( 'svc' in model_type and hasattr(clf, 'kernel')
                                                        and clf.kernel == 'linear'):
                explainer = shap.LinearExplainer(clf, X_scaled)
                shap_vals = explainer.shap_values(X_scaled)

                if isinstance(shap_vals, list) and len(shap_vals) > 1:
                    shap_vals = shap_vals[1]
            else:
                raise NotImplementedError(
                    f"SHAP explainer not set up for model type: {model_type}."
                    "Update function to handle your new classifier."
                )
            
            print(f"[DEBUG] {name} fold {i}: shap_vals shape {np.shape(shap_vals)}, X_scaled shape {np.shape(X_scaled)}")
           

            all_shap_values.append(shap_vals)
            all_feature_data.append(X_scaled)
            
        # concatenate over folds
        all_shap_values = np.vstack(all_shap_values)
        all_feature_data = np.vstack(all_feature_data)
        print(f"all_shap_vals shape: {all_shap_values.shape}")
        print(f"all_feature_data shape: {all_feature_data.shape}")

        # plot SHAP summary
        plt.figure(figsize = (8, 6))
        shap.summary_plot(
            all_shap_values,
            all_feature_data,
            feature_names = feature_names,
            show = False
        )

        plt.title(f"{name} SHAP Summary (Aggregated Across Folds)")
        plt.xlabel("SHAP Value (Impact on Predicting Diabetes)")
        plt.ylabel("Features")
        
        plt.tight_layout()
        plt.savefig(f"{outdir}/{name}_shap.png", dpi = 300, bbox_inches='tight')
        plt.close()

            
