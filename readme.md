# 🎓 Student Dropout Risk Predictor


## Overview
  Every year, many students leave school before finishing.  
  This project predicts **whether a student is at risk of dropping out** based on their grades, study time, absences, and other factors.  

  It was developed for the **Digitaley Drive AI/ML Hackathon (2025)**.  
  The aim is to give teachers and schools a simple tool to identify at-risk students early.

---

## 🚀 Live Demo
 [Try the app here](https://tram-brain-byte-hackathon-2025.streamlit.app/)

---

## Key Features
- Predicts dropout risk from academic and background data  
- Provides a **risk probability score** (0.0 → 1.0)  
- Identifies **important features** influencing risk (grades, absences, failures)  
- Interactive Streamlit app for real-time predictions  

---

## Workflow
  We followed a structured process, committing progress step by step:

  1. **Data Setup**  
    - Collected dataset (*Student Alcohol Consumption, UCI/Kaggle*)  
    - Organized repo with folders for `data/`, `src/`, `model/`, `app/`

  2. **Preprocessing**  
    - Cleaned dataset  
    - Engineered new features:  
      - `avg_grade` → mean of G1, G2, G3  
      - `grade_trend` → final grade minus first grade  
      - `passed_final` → 1 if final grade ≥ 10, else 0  
    - Encoded categorical features (e.g., `sex` → `sex_M`)  

  3. **Baseline Models**  
    - Logistic Regression  
    - Random Forest (default settings)  
    - Evaluated with accuracy, precision, recall, F1  

  4. **Improved Model**  
    - Random Forest with GridSearchCV (hyperparameter tuning)  
    - Cross-validation for robustness  
    - Evaluated with ROC-AUC and confusion matrix  
    - Extracted **feature importance**  

  5. **Deployment**  
    - Saved trained model, scaler, and feature names with `joblib`  
    - Built Streamlit app for predictions  
    - Deployed app on Streamlit Cloud  

  6. **Final Polish**  
    - README with demo link, results, and instructions  
    - Added docs with ROC Curve & Feature Importance  

## Dataset
  We are using the **Student Performance Dataset** from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/student+performance) (also available on Kaggle).
  It includes:
  - Demographics (age, gender, parental education)
  - Academic info (grades, study time)
  - Attendance (absences)
  - Lifestyle & family info

  The target variable is **dropout risk**, derived from academic & attendance indicators.

##  Project Structure
    Hackathon/
      │── app/ # Streamlit app
      │ └── app.py
      │── data/ # Raw & processed data
      │── model/ # Saved models & scaler
      │── notebooks/ # EDA and experiments
      │── src/ # Training & preprocessing scripts
      │ └── train_tuned.py
      │── requirements.txt # Dependencies
      │── README.md # Project documentation
      │── NOTES.md # Beginner-friendly explanation


## 🚀 Installation and training
  git clone https://github.com/<your-username>/student_dropout_predictor.git
  cd student_dropout_predictor
  pip install -r requirements.txt
  python src/train_tuned.py
