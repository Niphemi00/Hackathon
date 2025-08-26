from pathlib import Path

import joblib
import pandas as pd
from flasgger import Swagger
from flask import Flask, g, jsonify, request

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model/best_model.joblib"
SCALER_PATH = BASE_DIR / "model/scaler.joblib"
FEATURES_PATH = BASE_DIR / "model/feature_names.joblib"


def create_app():
    if (
        not MODEL_PATH.exists()
        or not SCALER_PATH.exists()
        or not FEATURES_PATH.exists()
    ):
        raise Exception(
            "Model/scaler/feature_names not found. Please ensure they are in /model folder."
        )

    app = Flask(__name__)
    swagger = Swagger(app)

    @app.post("/predict")
    def predict_score():
        """Make a prediction based on parameters
        ---
        parameters:
          - name: body
            in: body
            schema:
                type: object
                properties:
                    age:
                        type: integer
                        required: true
                    sex:
                        type: string
                        required: true
                        default: M
                        enum: [M, F]
                    study_time:
                        type: integer
                        required: true
                    past_failures:
                        type: integer
                        required: true
                    absences:
                        type: integer
                        required: true
                    grade_1:
                        type: integer
                        required: true
                        minimum: 0
                        maximum: 20
                    grade_2:
                        type: integer
                        required: true
                        minimum: 0
                        maximum: 20
                    final_grade:
                        type: integer
                        required: true
                        minimum: 0
                        maximum: 20
        responses:
            200:
                description: prediction result
                schema:
                type: object
                properties:
                    probablility:
                        type: integer
                    risk_score:
                        type: integer
                    advise:
                        type: string
        """
        try:
            if not request.is_json:
                return jsonify({"error": "Bad request"}), 400

            j_data = request.get_json()

            data = {
                "age": [j_data["age"]],
                "sex": [j_data["sex"]],
                "studytime": [j_data["study_time"]],
                "failures": [j_data["past_failures"]],
                "absences": [j_data["absences"]],
                "G1": [j_data["grade_1"]],
                "G2": [j_data["grade_2"]],
                "G3": [j_data["final_grade"]],
            }

            prob = predictor(data)
            risk_score = round(prob * 15)  # Risk score is out of 15
            advice = generate_advice(risk_score)

            return (
                jsonify(
                    {"probability": prob, "risk_score": risk_score, "advise": advice}
                ),  # Advise should be dynamically generated
                200,
            )
        except Exception as e:
            return jsonify({"error": str(e)})

    return app


def load_lib():
    model = joblib.load(MODEL_PATH)
    scalar = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    return model, scalar, feature_names


def predictor(input_dict: dict):
    """Predicts the probability of failure for the student from data provided."""
    model, scaler, feature_names = load_lib()
    if model is None or scaler is None or feature_names is None:
        raise ValueError("models never loaded, cannot make predictions.")

    # Construct input dataframe
    input_df = pd.DataFrame(input_dict)

    # Feature engineering
    input_df["avg_grade"] = input_df[["G1", "G2", "G3"]].mean(axis=1)
    input_df["grade_trend"] = input_df["G3"] - input_df["G1"]
    input_df["passed_final"] = (input_df["G3"] >= 10).astype(int)

    # OHE
    input_df = pd.get_dummies(input_df, columns=["sex"], drop_first=True)

    # Ensure all expected features are present
    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0

    # Reorganuze columns to match training
    input_df = input_df[feature_names]

    # Scale numeric values
    input_scaled = scaler.transform(input_df)

    # make valid prediction
    prob = model.predict_proba(input_scaled)[:, 1][0]

    return prob


def generate_advice(score, max_score=15):
    avg = round(score / max_score)
    if avg < 1:
        return "Low risk assessment. Student demonstrates good academic performance, regular attendance, and positive engagement. Continue current support and maintain regular check-ins."
    else:
        return ""
