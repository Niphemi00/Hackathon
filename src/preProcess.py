"""
src/preprocess.py

Preprocessing, feature engineering, and target labeling for the Student Dropout project.

Usage:
    python src/preprocess.py \
        --mat data/student-mat.csv \
        --por data/student-por.csv \
        --combined data/combined_students_raw.csv \
        --output data/processed_students_for_model.csv \
        --quantile 0.8
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(mat_path: Path, por_path: Path, combined_path: Path = None):
    """
    Load either an existing combined file or read and concat the two CSVs.
    """
    if combined_path and combined_path.exists():
        print(f"Loading combined dataset from {combined_path}")
        df = pd.read_csv(combined_path)
    else:
        print(f"Loading separate datasets:\n - {mat_path}\n - {por_path}")
        df_mat = pd.read_csv(mat_path)
        df_por = pd.read_csv(por_path)
        df = pd.concat([df_mat, df_por], ignore_index=True)
        print(f"Combined shape: {df.shape}")
    return df

def engineer_labels(df: pd.DataFrame,
                    grade_thresh: int = 10,
                    absences_thresh: int = 10,
                    decline_thresh: int = -3,
                    quantile: float = 0.8):
    """
    Create two labels:
      - dropout_label_rule: interpretable rules
      - dropout_label_quantile: risk score -> top quantile flagged as 'at risk'
    Also create auxiliary features useful for modeling.
    """
    df = df.copy()

    # Basic features
    df['avg_grade'] = df[['G1', 'G2', 'G3']].mean(axis=1)
    df['grade_trend'] = df['G3'] - df['G1']           # positive = improved, negative = declined
    df['passed_final'] = (df['G3'] >= 10).astype(int) # 10/20 considered pass

    # Rule-based label (easy to explain to judges)
    # Conditions: low final grade OR many absences OR steep decline
    df['dropout_label_rule'] = (
        (df['G3'] < grade_thresh) |
        (df['absences'] >= absences_thresh) |
        (df['grade_trend'] <= decline_thresh)
    ).astype(int)

    # Quantile-based risk score (continuous) -> label by top quantile
    # Normalize components to [0,1]
    max_absences = max(df['absences'].max(), 1)
    grade_component = (20 - df['G3']) / 20  # higher when grade is low
    abs_component = df['absences'] / max_absences
    decline_component = np.maximum(0, df['G1'] - df['G3'])
    max_decline = max(decline_component.max(), 1)
    decline_component = decline_component / max_decline

    # Weights: grade matters most, absences next, sudden decline some weight.
    df['risk_score'] = 0.5 * grade_component + 0.4 * abs_component + 0.1 * decline_component

    threshold = df['risk_score'].quantile(quantile)
    df['dropout_label_quantile'] = (df['risk_score'] >= threshold).astype(int)

    # Diagnostics
    print("Label distribution (rule):")
    print(df['dropout_label_rule'].value_counts(normalize=True))
    print("\nLabel distribution (quantile):")
    print(df['dropout_label_quantile'].value_counts(normalize=True))
    print(f"\nQuantile threshold (q={quantile}): {threshold:.4f}")

    return df

def encode_and_scale(df: pd.DataFrame, scaler_path: Path = Path('model/scaler.joblib')):
    """
    - One-hot encode categorical variables (drop_first=True to reduce dim)
    - Scale numeric columns (StandardScaler)
    - Save the scaler so we can use the same transform during inference.
    """
    df = df.copy()
    # Keep the targets and risk_score aside
    targets = ['dropout_label_rule', 'dropout_label_quantile', 'risk_score']
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"Expected target column '{t}' in dataframe")

    # Identify categorical columns (object or category)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    # Numeric columns (we will scale these)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # exclude targets from scaling
    num_cols_to_scale = [c for c in num_cols if c not in targets]

    print(f"Numeric columns to scale ({len(num_cols_to_scale)}): {num_cols_to_scale}")

    # Scale numeric columns
    scaler = StandardScaler()
    df[num_cols_to_scale] = scaler.fit_transform(df[num_cols_to_scale])

    # Save scaler
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # One-hot encode categorical features (drop_first to avoid collinearities)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        print(f"After one-hot encoding, shape: {df.shape}")
    else:
        print("No categorical columns found for encoding.")

    return df, scaler

def save_processed(df: pd.DataFrame, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to {output_path}")

def main(args):
    mat = Path(args.mat)
    por = Path(args.por)
    combined = Path(args.combined) if args.combined else None
    out = Path(args.output)
    quantile = args.quantile

    df = load_data(mat, por, combined if combined and combined.exists() else None)
    df = engineer_labels(df, grade_thresh=args.grade_thresh,
                         absences_thresh=args.absences_thresh,
                         decline_thresh=args.decline_thresh,
                         quantile=quantile)
    df_encoded, scaler = encode_and_scale(df, scaler_path=Path('model/scaler.joblib'))
    save_processed(df_encoded, out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat', type=str, default='data/student-mat.csv')
    parser.add_argument('--por', type=str, default='data/student-por.csv')
    parser.add_argument('--combined', type=str, default='data/combined_students_raw.csv')
    parser.add_argument('--output', type=str, default='data/processed_students_for_model.csv')
    parser.add_argument('--quantile', type=float, default=0.80, help='Top quantile to mark as high-risk')
    parser.add_argument('--grade_thresh', type=int, default=10, help='Grade threshold for rule label')
    parser.add_argument('--absences_thresh', type=int, default=10, help='Absences threshold for rule label')
    parser.add_argument('--decline_thresh', type=int, default=-3, help='Grade trend threshold (G3-G1) for rule label')
    args = parser.parse_args()
    main(args)
