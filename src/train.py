"""
src/train.py

Train baseline models (Logistic Regression, Random Forest) to predict dropout risk.

Usage:
    python src/train.py \
        --data data/processed_students_for_model.csv \
        --target dropout_label_quantile \
        --output model/model.joblib
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def load_data(path: Path, target: str):
    df = pd.read_csv(path)
    X = df.drop(columns=[target])  # features
    y = df[target]                 # label
    return X, y

def evaluate_model(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {auc:.4f}")

def main(args):
    X, y = load_data(Path(args.data), args.target)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    evaluate_model(log_reg, X_test, y_test, "Logistic Regression")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    evaluate_model(rf, X_test, y_test, "Random Forest")

    # Choose best model :by F1-score or ROC-AUC (For now, save Random Forest (commonly better))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, args.output)
    print(f"\n✅ Model saved to {args.output}")
    
    importances = rf.feature_importances_
    feature_names = X_train.columns
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    print("\n=== Top 10 Features Influencing Dropout Risk ===")
    print(feat_imp.head(10))
    #Save file for reference
    feat_imp.to_csv("model/feature_importance.csv")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed_students_for_model.csv")
    parser.add_argument("--target", type=str, default="dropout_label_quantile")
    parser.add_argument("--output", type=str, default="model/model.joblib")
    args = parser.parse_args()
    main(args)
