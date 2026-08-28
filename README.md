# 🎓 Student Performance Analytics

## 📌 Project Overview

Student Performance Analytics is a Data Science and Machine Learning application designed to analyze student academic performance, identify important performance factors, visualize data-driven insights, and predict a student's expected final score.

The project combines **Exploratory Data Analysis (EDA), Data Visualization, Statistical Analysis, and Machine Learning** into an interactive Streamlit dashboard.

---

## 🎯 Project Objectives

* Analyze student academic performance
* Perform exploratory data analysis
* Identify relationships between different performance factors
* Visualize important patterns and trends
* Compare machine learning models
* Predict a student's expected final score
* Provide an interactive analytics dashboard

---

## 📊 Dataset

The dataset contains **5,000 student records** and includes the following features:

| Feature         | Description                   |
| --------------- | ----------------------------- |
| StudyHours      | Average study hours per day   |
| Attendance      | Student attendance percentage |
| PreviousScore   | Previous academic score       |
| AssignmentScore | Assignment performance        |
| SleepHours      | Average sleep hours per day   |
| FinalScore      | Final academic score          |

---

## 🔎 Data Analysis

The project performs:

* Dataset exploration
* Missing-value analysis
* Duplicate-value analysis
* Statistical summary
* Correlation analysis
* Final-score distribution
* Performance categorization
* Study-hours vs final-score analysis
* Feature relationship analysis

---

## 🤖 Machine Learning

Two regression models were evaluated:

* Linear Regression
* Random Forest Regression

Based on the evaluation results, **Linear Regression** was selected as the final model.

### Model Evaluation

| Model             |  MAE |   MSE | R² Score |
| ----------------- | ---: | ----: | -------: |
| Linear Regression | 3.39 | 18.54 |    0.872 |
| Random Forest     | 3.68 | 22.12 |    0.847 |

The Linear Regression model achieved an R² score of approximately **0.872** on the test dataset.

---

## 📈 Streamlit Application

The application contains three major sections:

### 1. 📊 Analytics Dashboard

Provides:

* Total student count
* Average final score
* Average attendance
* Average study hours
* Final-score distribution
* Performance categories
* Study-hours vs final-score visualization
* Correlation heatmap
* Key data insights

### 2. 🔮 Student Prediction

Users can enter:

* Study hours
* Attendance
* Previous score
* Assignment score
* Sleep hours

The application then predicts the student's expected final score.

### 3. 📋 Data Explorer

Allows users to:

* View the dataset
* Filter students by final score
* View statistical summaries
* Check missing values
* Explore the dataset interactively

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

## 📁 Project Structure

```text
Student performance prediction system/
│
├── app.py
├── requirements.txt
├── README.md
│
├── analytics/
│   ├── analysis.py
│   └── student_performance.csv
│
├── models/
│   └── student_model.pkl
│
└── project_env/
```

---

## ▶️ How to Run the Project

### Step 1: Activate the environment

```powershell
.\project_env\Scripts\Activate.ps1
```

### Step 2: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 3: Run the Streamlit application

```powershell
python -m streamlit run app.py
```

### Step 4: Open the application

Open the local Streamlit URL shown in the terminal, usually:

```text
http://localhost:8501
```

---

## 📌 Project Workflow

```text
Student Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Statistical Analysis
       ↓
Data Visualization
       ↓
Correlation Analysis
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Selection
       ↓
Streamlit Dashboard
       ↓
Student Score Prediction
```

---

## 👩‍💻 Project Focus

This project focuses primarily on **Data Science and Analytics**, with Machine Learning used as one component of the overall solution.

The main objective is to transform raw student data into meaningful insights and provide an interactive platform for performance analysis and prediction.

