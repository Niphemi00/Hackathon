
# Learning Notes – Student Dropout Risk Predictor
Developed by Joshua Ikem for the Digitaley Drive AI/ML Hackathon (2025).

This file explains the whole project in **simple words**.  
It covers **what we did, why we did it, and what the code means**.

## 1. The Problem
We want to know if a student will **drop out** or **stay in school**.  
We only have some information about them:
- Age  
- Gender (boy or girl)  
- Study time per week  
- Past failures  
- Absences  
- Grades (G1, G2, G3)

## 2. The Plan
We followed 6 simple steps:

1. **Get the data** → download student dataset  
2. **Clean the data** → fix missing values, add useful columns  
3. **Train models** → teach a computer how to guess dropout risk  
4. **Improve models** → make the guesses smarter  
5. **Save the model** → freeze it so we can use it later  
6. **Build an app** → let anyone enter student info and get a prediction  


## 3. Data Preprocessing
We prepared the data like this:

- Made new columns:
  - `avg_grade` = average of G1, G2, G3  
  - `grade_trend` = G3 – G1 (did grades go up or down?)  
  - `passed_final` = 1 if G3 ≥ 10, else 0  

- Turned "sex" into numbers:
  - "M" → `sex_M = 1`  
  - "F" → `sex_M = 0`  

This way, the computer can understand it.

## 4. Models
We tried two main models:

- **Logistic Regression**:  
  Simple model, like flipping a weighted coin.  

- **Random Forest**:  
  Smarter model made of many "trees". Each tree makes a guess, and the forest votes.  
  We found Random Forest was better, so we improved it.

---

## 5. Model Improvement
We used **GridSearchCV** to test many Random Forest settings:
- Number of trees (`n_estimators`)  
- How deep the trees go (`max_depth`)  
- Minimum samples to split nodes (`min_samples_split`)  

We picked the settings that worked best.  
Then we checked results with:
- **Confusion Matrix** (how many right/wrong guesses)  
- **ROC-AUC** (how good the model is overall)  
- **Feature Importance** (which features matter most)  

---

## 6. Saving the Model
We saved 3 things with `joblib`:
- `best_model.joblib` → the trained Random Forest  
- `scaler.joblib` → makes numbers consistent  
- `feature_names.joblib` → remembers which columns we used  

---

## 7. The App
We built a **Streamlit app**.  
Steps inside the app:
1. User enters student details (age, grades, absences, etc.)  
2. App builds a row of data  
3. App makes sure it matches training columns  
4. App scales the numbers  
5. Model predicts **dropout risk probability**  
6. App shows result:
   - Not at Risk  
   - At Risk  


## 8. Why This Is Cool
- Teachers can test "what-if" scenarios:  
  > "If the student studies more, how does risk change?"  
- Schools can identify struggling students earlier.  
- Judges can play with the live demo instantly.  


## 9. Code in Simple Words
  ### Training Script (`train_tuned.py`)
  - Load the data  
  - Split into training/testing sets  
  - Train Logistic Regression (baseline)  
  - Train Random Forest with GridSearchCV  
  - Pick the best model  
  - Save model, scaler, and feature names  
  ### App (`app.py`)
  - Load the saved model, scaler, and features  
  - Take user input from a form  
  - Create a row of data with the same features as training  
  - Scale it  
  - Predict dropout risk  
  - Show result nicely in the browser  

That’s it.  
We turned raw student data → into a working AI model → into a live app.
