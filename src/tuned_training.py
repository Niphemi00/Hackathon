"""
src/train_tuned.py

Train an improved model with hyperparameter tuning and cross-validation.

Usage:
    python src/train_tuned.py \
        --data data/processed_students_for_model.csv \
        --target dropout_label_quantile \
        --output model/best_model.joblib
"""

import argparse
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def main(args):
    # Load dataset
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Logistic Regression (baseline) ---
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    print("\n=== Logistic Regression ===")
    print(classification_report(y_test, y_pred_lr))

    # --- Random Forest with GridSearch ---
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    }
    grid = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    print("\n=== Random Forest (Grid Search) ===")
    print("Best Parameters:", grid.best_params_)
    best_rf = grid.best_estimator_

    y_pred_rf = best_rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

    if hasattr(best_rf, "predict_proba"):
        y_prob_rf = best_rf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob_rf)
        print(f"ROC-AUC: {auc:.4f}")

    # --- Save model, scaler, and feature names ---
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(best_rf, args.output)

    # Save feature names (column order matters!)
    feature_names_path = Path(args.output).parent / "feature_names.joblib"
    joblib.dump(X.columns.tolist(), feature_names_path)

    # Save scaler
    scaler = StandardScaler()
    scaler.fit(X)  # fit on full data
    scaler_path = Path(args.output).parent / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    print(f"\nModel saved to {args.output}")
    print(f"Feature names saved to {feature_names_path}")
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed_students_for_model.csv")
    parser.add_argument("--target", type=str, default="dropout_label_quantile")
    parser.add_argument("--output", type=str, default="model/best_model.joblib")
    args = parser.parse_args()
    main(args)
