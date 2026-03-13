# CodeAlpha_Creditworthiness_Prediction_System
📋 Project Overview
An end-to-end Machine Learning Web Application that predicts creditworthiness based on multiple financial and demographic factors. This project demonstrates the complete ML lifecycle from data preprocessing to model deployment.
🎯 Business Problem
Financial institutions need to assess loan applicants quickly and accurately. Traditional manual assessment is time-consuming and prone to bias. This system provides:
Instant credit decisions (seconds vs. days)
Data-driven risk assessment
Consistent evaluation criteria
Reduced human bias
✨ Key Features
Feature
Description
🤖 ML Model
Random Forest Classifier with class balancing
🌐 Web Interface
Clean, responsive Flask-based UI
📊 13 Input Features
Income, Debt, Credit Score, Employment, etc.
⚡ Real-time Prediction
Instant creditworthiness assessment
🎨 Modern UI
Professional design with validation
📱 Responsive
Works on desktop and mobile

📊 Dataset Information
Attribute
Details
Records
12,000 applicants
Features
13 input variables
Target
Creditworthiness (0/1)
Class Distribution
~70% Creditworthy, ~30% Not
Input Features
Age - Applicant age (18-100)
Gender - Male/Female
Education - High School, Bachelor, Master, PhD
Income - Annual income ($)
Debt - Total existing debt ($)
Credit Score - Credit rating (300-850)
Loan Amount - Requested loan amount ($)
Loan Term - Repayment period (months)
Num Credit Cards - Number of credit cards
Payment History - Good/Average/Bad
Employment Status - Employed/Unemployed/Self-Employed
Residence Type - Owned/Rented/Mortgaged
Marital Status - Single/Married/Divorced
🎯 Model Performance
Metric
Score
Accuracy
~70%
Precision
~0.82
Recall
~0.83
F1-Score
~0.82
ROC-AUC
~0.50
⚠️ Note: The model shows moderate performance. In production, further feature engineering and hyperparameter tuning would be recommended.
💻 Usage Guide
Making a Prediction
Open the application in your browser
Fill in all required fields in the form
Click "Predict Creditworthiness"
View instant results with confidence score
