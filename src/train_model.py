"""
Model Training & Evaluation Pipeline for Customer Churn Prediction.

Trains, cross-validates, and evaluates:
1. Logistic Regression (interpretable baseline)
2. Random Forest (nonlinear ensemble)
3. XGBoost (gradient boosted decision trees)

Evaluates on Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix.
Selects the best model based on business relevance (ROC-AUC & Recall).
Saves models, preprocessor, and complete evaluation artifacts for the Streamlit dashboard.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from src.preprocessing import (
    load_raw_data,
    clean_dataset,
    prepare_training_data,
    get_feature_preprocessor,
    extract_feature_names,
    ID_COL,
    TARGET_COL
)


def train_and_evaluate_all_models(data_path: str = "data/customer_churn.csv", output_dir: str = "models"):
    """
    Main training workflow:
    - Loads and cleans raw data
    - Performs stratified train-test split (80/20)
    - Fits preprocessor strictly on training data
    - Trains Logistic Regression, Random Forest, and XGBoost
    - Computes 5-fold cross-validation and test set evaluation metrics
    - Generates ROC curves, PR curves, and feature importances
    - Serializes models and comparison metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading dataset from: {data_path}")
    raw_df = load_raw_data(data_path)
    clean_df = clean_dataset(raw_df)
    
    # Store customerIDs for linking test set predictions
    customer_ids = clean_df[ID_COL] if ID_COL in clean_df.columns else pd.Series(range(len(clean_df)))
    
    X, y = prepare_training_data(clean_df)
    
    # Stratified train/test split to preserve churn ratio
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, customer_ids, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"Train Churn Rate: {y_train.mean():.2%} | Test Churn Rate: {y_test.mean():.2%}")
    
    # Fit preprocessor strictly on training data
    print("Fitting ColumnTransformer preprocessor on X_train...")
    preprocessor = get_feature_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_test_trans = preprocessor.transform(X_test)
    
    feature_names = extract_feature_names(preprocessor)
    print(f"Extracted {len(feature_names)} features after One-Hot Encoding and Scaling.")
    
    # Calculate scale_pos_weight for XGBoost to balance positive class
    pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    
    # Define candidate models
    models_dict = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    test_preds_df = pd.DataFrame({
        "customerID": ids_test.values,
        "Actual_Churn": y_test.values
    })
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models_dict.items():
        print(f"\n--- Training {name} ---")
        
        # 5-Fold Stratified Cross-Validation on Train Set
        cv_scores = cross_validate(
            model,
            X_train_trans,
            y_train,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1
        )
        
        # Train on full train set
        model.fit(X_train_trans, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test_trans)
        y_prob = model.predict_proba(X_test_trans)[:, 1]
        
        # Calculate metrics
        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_prob))
        cm = confusion_matrix(y_test, y_pred).tolist()
        
        # Specificity: True Negative Rate
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        
        # Compute ROC Curve points (downsampled for clean JSON serialization)
        fpr, tpr, roc_thresh = roc_curve(y_test, y_prob)
        step = max(1, len(fpr) // 50)
        roc_data = {
            "fpr": fpr[::step].tolist() + [float(fpr[-1])],
            "tpr": tpr[::step].tolist() + [float(tpr[-1])],
            "thresholds": roc_thresh[::step].tolist() + [float(roc_thresh[-1])]
        }
        
        # Precision-Recall Curve
        pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)
        pr_step = max(1, len(pr_prec) // 50)
        pr_data = {
            "precision": pr_prec[::pr_step].tolist() + [float(pr_prec[-1])],
            "recall": pr_rec[::pr_step].tolist() + [float(pr_rec[-1])]
        }
        
        # Feature importances / coefficients
        feature_importance_dict = {}
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for idx in sorted_idx[:15]:
                feature_importance_dict[feature_names[idx]] = float(importances[idx])
        elif hasattr(model, "coef_"):
            coefs = model.coef_[0]
            sorted_idx = np.argsort(np.abs(coefs))[::-1]
            for idx in sorted_idx[:15]:
                feature_importance_dict[feature_names[idx]] = float(coefs[idx])
                
        # Store results
        results[name] = {
            "test_metrics": {
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "roc_auc": round(roc_auc, 4),
                "specificity": round(specificity, 4)
            },
            "cv_metrics": {
                "cv_accuracy_mean": round(float(cv_scores["test_accuracy"].mean()), 4),
                "cv_accuracy_std": round(float(cv_scores["test_accuracy"].std()), 4),
                "cv_recall_mean": round(float(cv_scores["test_recall"].mean()), 4),
                "cv_recall_std": round(float(cv_scores["test_recall"].std()), 4),
                "cv_f1_mean": round(float(cv_scores["test_f1"].mean()), 4),
                "cv_f1_std": round(float(cv_scores["test_f1"].std()), 4),
                "cv_roc_auc_mean": round(float(cv_scores["test_roc_auc"].mean()), 4),
                "cv_roc_auc_std": round(float(cv_scores["test_roc_auc"].std()), 4),
            },
            "confusion_matrix": cm,
            "roc_curve": roc_data,
            "pr_curve": pr_data,
            "top_features": feature_importance_dict
        }
        
        # Add predictions to dataframe
        test_preds_df[f"{name}_Prob"] = np.round(y_prob, 4)
        test_preds_df[f"{name}_Pred"] = y_pred
        
        # Save individual model
        slug = name.lower().replace(" ", "_")
        model_filename = os.path.join(output_dir, f"{slug}.pkl")
        joblib.dump(model, model_filename)
        print(f"Saved {name} to {model_filename}")
        print(f"Test Accuracy: {acc:.2%} | Recall: {rec:.2%} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    # Select Best Model based on ROC-AUC and F1-Score
    best_model_name = max(results.keys(), key=lambda k: results[k]["test_metrics"]["roc_auc"] + results[k]["test_metrics"]["f1_score"])
    print(f"\n==========================================")
    print(f"Selected Best Model: {best_model_name}")
    print(f"ROC-AUC: {results[best_model_name]['test_metrics']['roc_auc']:.4f}")
    print(f"F1-Score: {results[best_model_name]['test_metrics']['f1_score']:.4f}")
    print(f"Recall: {results[best_model_name]['test_metrics']['recall']:.4f}")
    print(f"==========================================")
    
    # Save the selected best model as churn_model.pkl
    best_model_slug = best_model_name.lower().replace(" ", "_")
    best_model = joblib.load(os.path.join(output_dir, f"{best_model_slug}.pkl"))
    joblib.dump(best_model, os.path.join(output_dir, "churn_model.pkl"))
    print(f"Serialized winning model to {os.path.join(output_dir, 'churn_model.pkl')}")
    
    # Save fitted preprocessor
    preprocessor_path = os.path.join(output_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}")
    
    # Save model comparison json
    comparison_path = os.path.join(output_dir, "model_comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved comparison metrics to {comparison_path}")
    
    # Save metadata
    metadata = {
        "best_model": best_model_name,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "test_size": 0.20,
        "random_state": 42,
        "thresholds": {
            "low_risk_max": 0.35,
            "high_risk_min": 0.65,
            "default_decision": 0.50
        },
        "target_distribution": {
            "total": int(len(clean_df)),
            "retained": int((clean_df[TARGET_COL] == "No").sum()),
            "churned": int((clean_df[TARGET_COL] == "Yes").sum()),
            "churn_rate": round(float((clean_df[TARGET_COL] == "Yes").mean()), 4)
        }
    }
    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to {metadata_path}")
    
    # Append input features to test_preds_df for rich exploration in Streamlit
    for col in X_test.columns:
        test_preds_df[col] = X_test[col].values
        
    test_preds_path = os.path.join(output_dir, "test_predictions.csv")
    test_preds_df.to_csv(test_preds_path, index=False)
    print(f"Saved test predictions table to {test_preds_path}")
    
    print("\nTraining workflow completed successfully!")
    return results, metadata


if __name__ == "__main__":
    train_and_evaluate_all_models()
