# 📊 Customer Churn Prediction & Retention Analytics System

A machine learning-based web application that predicts whether a customer is likely to churn and helps businesses identify high-risk customers for targeted retention strategies.

The project uses **Python, Scikit-learn, XGBoost, Pandas, Plotly, and Streamlit** to build an end-to-end customer churn prediction system.

---

## 📌 Project Overview

Customer churn occurs when a customer stops using a company's product or service.

The goal of this project is to:

- Predict customers who are likely to churn
- Identify factors associated with customer churn
- Compare multiple machine learning models
- Provide customer-level churn predictions
- Display results through an interactive Streamlit dashboard
- Support data-driven customer retention decisions

---

## 🎯 Objectives

1. Perform data preprocessing and cleaning.
2. Analyze customer characteristics and churn patterns.
3. Train multiple machine learning classification models.
4. Evaluate and compare model performance.
5. Predict churn probability for individual customers.
6. Identify high-risk customers.
7. Provide an interactive dashboard for analysis.

---

## ✨ Key Features

### 🔹 Data Preprocessing
- Missing-value handling
- Categorical variable encoding
- Numerical feature processing
- Feature transformation
- Train-test split

### 🔹 Exploratory Data Analysis
- Customer demographic analysis
- Tenure analysis
- Contract analysis
- Payment method analysis
- Churn distribution
- Service usage analysis

### 🔹 Machine Learning Models

The project compares:

- Logistic Regression
- Random Forest
- XGBoost

### 🔹 Churn Prediction

The system provides:

- Churn prediction
- Churn probability
- Customer risk identification
- Model-based prediction results

### 🔹 Interactive Dashboard

Built using **Streamlit** and **Plotly**, the dashboard provides:

- Dataset overview
- Churn statistics
- Model comparison
- Customer prediction
- Interactive visualizations
- High-risk customer analysis

---

## 📂 Dataset

The project uses the **IBM Telco Customer Churn dataset**.

The dataset contains customer information such as:

- Customer demographics
- Gender
- Senior citizen status
- Partner/dependent information
- Tenure
- Internet service
- Phone service
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

### Target Variable

**Churn**

- `Yes` → Customer churned
- `No` → Customer did not churn

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| XGBoost | Gradient boosting model |
| Plotly | Interactive visualizations |
| Streamlit | Web application |
| Joblib | Model serialization |
| Jupyter Notebook | Data analysis and experimentation |
| Git & GitHub | Version control |

---

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Data Preprocessing
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Prediction
   ↓
Streamlit Dashboard