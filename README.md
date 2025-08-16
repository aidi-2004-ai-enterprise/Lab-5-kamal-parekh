# Lab-5 Company Bankruptcy Prediction Pipeline

## Overview
This project implements a **Model Implementation and Evaluation Pipeline** for predicting company bankruptcy using the [Company Bankruptcy Prediction dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction). The pipeline automates data preprocessing, feature selection, model training, hyperparameter tuning, evaluation, and interpretability analysis. It is designed to be reproducible and ready for deployment in production environments, including AirFlow DAGs.

## Features
- **Data Preprocessing**: Handles missing values, normalization, encoding, and outlier treatment.  
- **Feature Selection**: Reduces noise using correlation filtering and feature importance techniques.  
- **Model Training & Tuning**: Trains Logistic Regression, Random Forest, and XGBoost models with cross-validation and hyperparameter tuning.  
- **Model Evaluation**: Generates ROC-AUC curves, Brier scores, calibration plots, and F1 scores for train/test sets.  
- **Interpretability**: Provides SHAP value analysis for feature contributions.  
- **Data Stability Analysis**: Computes Population Stability Index (PSI) to detect data drift.  
- **Reproducibility**: Random seeds are set; PEP-compliant code using Ruff.

python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

##How to Run

pip install -r requirements.txt
python training_pipeline.py --input_csv "path/to/data.csv" --target_column "Bankrupt?"

##Notes

Ensure your CSV data path and target column are correct.

The pipeline supports stratified train/test splits to handle class imbalance.

Ruff was used to ensure PEP-compliance; run ruff check training_pipeline.py to verify code quality.

The pipeline is designed for reproducibility; random seeds are fixed.




