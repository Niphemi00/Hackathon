# Student Dropout Risk Predictor (Hackathon Project August 2025)

## 📌 Overview
  This project is being developed for the **Digitaley Drive AI/ML Hackathon**.
  It predicts the likelihood of a student dropping out based on academic,
  attendance, and socioeconomic factors.

  The aim is to help schools, policymakers, and NGOs identify at-risk students early
  and take preventive measures.

## 📂 Project Structure
  Hackathon/
  │── data/ ==>> would contain the Raw and processed data
  │── notebooks/ ==>> would contain the EDA and model development
  │── src/ ==>> would contain the Scripts for training and inference
  │── app/ ==>> would contain the Streamlit web app for uploading data and getting predictions
  │── model/ ==>> would contain the Saved model artifacts
  │── requirements.txt ==>> would contain the neccesary Python dependencies
  │── README.md

## 📊 Dataset
  We are using the **Student Alcohol Consumption Dataset** from Kaggle:
  https://www.kaggle.com/datasets/uciml/student-alcohol-consumption
  This dataset contains information about students' alcohol consumption, academic performance, attendance, and socioeconomic factors.

  This dataset comprises of three files:
  - `student-mat.csv` – Data on students in the mathematics course.
  - `student-por.csv` – Data on students in the Portuguese language course.

  It includes:
  - Demographics (age, gender, parental education)
  - Academic info (grades: G1, G2, G3)
  - Attendance (absences)
  - Lifestyle & family info
  - Alcohol consumption habits

## File download and setup
  git clone https://github.com/Niphemi00/Hackathon.git
  cd Hackathon
  pip install -r requirements.txt

We will engineer a **dropout risk label** using low grades & high absences.


## Dataset setup instructions for you
  # 1. Install Kaggle CLI
  pip install kaggle

  # 2. Place your Kaggle API token (kaggle.json) in:
  # Linux/Mac: ~/.kaggle/kaggle.json
  # Windows: C:\Users\<username>\.kaggle\kaggle.json

  # 3. Download the dataset
  kaggle datasets download -d uciml/student-alcohol-consumption -p data/

  # 4. Unzip
  unzip data/student-alcohol-consumption.zip -d data/
