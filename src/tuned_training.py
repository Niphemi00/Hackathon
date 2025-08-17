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

    # --- Logistic Regression ---
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

    # Save best model
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, args.output)
    print(f"\n Best model saved to {args.output}")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Simulated ground truth (y_true) and predictions (y_pred)
y_true = np.array([0]*90 + [1]*60)   # 90 not at risk, 60 at risk
y_pred = np.array([0]*82 + [1]*8 + [0]*15 + [1]*45)  # simulated model predictions

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not at Risk","At Risk"],
            yticklabels=["Not at Risk","At Risk"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix Example")
plt.show()

# --- ROC Curve ---
# Simulated prediction probabilities (y_scores)
y_scores = np.concatenate([np.random.rand(90)*0.4, np.random.rand(60)*0.9+0.1])  # lower for safe, higher for risk
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", lw=2)
plt.plot([0,1], [0,1], linestyle="--", color="grey")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Example")
plt.legend()
plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/processed_students_for_model.csv")
    parser.add_argument("--target", type=str, default="dropout_label_quantile")
    parser.add_argument("--output", type=str, default="model/best_model.joblib")
    args = parser.parse_args()
    main(args)
