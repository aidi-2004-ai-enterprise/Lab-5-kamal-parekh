# Lab 5: Model Implementation and Evaluation Pipeline

## 1. Introduction
Objective: Train, evaluate, and compare three models (Logistic Regression, Random Forest, XGBoost) on the Company Bankruptcy Prediction dataset to recommend the best model for deployment.

---

## 2. Exploratory Data Analysis (EDA)
- Feature distributions examined using histograms and boxplots.
- Correlation heatmap identified multicollinearity between features X and Y.
- Outliers detected in feature Z; addressed during preprocessing.
- Missing values handled by median imputation

---

## 3. Data Preprocessing
- Normalized numeric features using MinMaxScaler.
- Categorical variables encoded using one-hot encoding.
- Train/test split: 80/20 stratified by target.
- PSI calculated to ensure no significant train-test drift.

---

## 4. Feature Selection
- Selected features based on XGBoost importance and correlation filtering.
- Logistic Regression   
- Random Forest  
- XGBoost  
- Multicollinearity addressed by removing highly correlated features (corr > 0.9).

---

## 5. Hyperparameter Tuning
| Model | Hyperparameters | Best Parameters |
|-------|----------------|----------------|
| Logistic Regression | solver, penalty, C | solver='lbfgs', penalty='l2', C=0.001 |
| Random Forest | n_estimators, max_depth, min_samples_split | n_estimators=200, max_depth=5, min_samples_split=10 |
| XGBoost | max_depth, learning_rate, n_estimators | max_depth=4, learning_rate=0.1, n_estimators=150 |

---

## 6. Model Training & Evaluation
| Model | ROC-AUC Train | ROC-AUC Test | F1 Train | F1 Test | Brier Train | Brier Test |
|-------|---------------|--------------|----------|---------|-------------|------------|
| Logistic Regression | 0.939 | 0.888 | 0.305 | 0.294 | 0.106 | 0.106 |
| Random Forest | 0.980 | 0.946 | 0.481 | 0.398 | 0.050 | 0.054 |
| XGBoost | 0.975 | 0.948 | 0.470 | 0.400 | 0.052 | 0.053 |

*ROC Curve Example:*  
![XGBoost ROC]
<img width="960" height="720" alt="RandomForestClassifier_calibration" src="https://github.com/user-attachments/assets/1462b7ee-0996-4e97-b960-9208c38a0f50" />


*Calibration Curve Example:*  
![XGBoost Calibration]
<img width="960" height="720" alt="RandomForestClassifier_calibration" src="https://github.com/user-attachments/assets/5db03cb5-f238-4970-9ad3-22dfd82acae7" />


- Random Forest & XGBoost show slightly better test performance than Logistic Regression.
- Minimal overfitting observed.

---

## 7. Interpretability (SHAP)
- SHAP summary plot for XGBoost:
![SHAP Summary]
- Key features impacting bankruptcy prediction: Operating Profit, Cash Flow Rate, Total Assets.
<img width="1200" height="2962" alt="psi_bar" src="https://github.com/user-attachments/assets/6abf7f6b-d6ac-4fe7-bf72-0faecc85843b" />

---

## 8. Population Stability Index (PSI)
| Feature | PSI Value |
|---------|-----------|
| Operating Profit per Person | 0.020 |
| Working Capital to Total Assets | 0.020 |
| Cash Flow Rate | 0.019 |
- All PSI values < 0.25 → no significant drift detected.

---

## 9. Challenges and Reflections
- Challenge: Hyperparameter tuning was computationally intensive.  
  - Solution: Limited grid search candidates.  
- Challenge: Handling multicollinearity.  
  - Solution: Correlation-based feature removal.  
- Challenge: Interpreting feature importance across models.  
  - Solution: SHAP used for XGBoost for better interpretability.

---

## 10. Conclusion & Deployment Recommendation
- XGBoost achieved the best test performance (ROC-AUC 0.948, F1 0.400) with robust calibration.
- Recommended for deployment due to strong generalization and interpretability via SHAP.

ruff linting 
- Purpose: Ensure code quality, enforce PEP standards, improve readability and maintainability.
<img width="788" height="66" alt="image" src="https://github.com/user-attachments/assets/fb1fc1bb-3fd3-462b-bf0f-c7b6a676a3c1" />



1. Main Challenges and How They Were Overcome

Class Imbalance and Skewed Features: Some features and the target were imbalanced. Addressed by using stratified train/test splits and scaling features appropriately.

Multicollinearity: High correlation between features affected model interpretability. Solved using correlation filtering and feature selection techniques.

Hyperparameter Tuning Efficiency: Large search space made tuning computationally expensive. Resolved by using limited grid search with cross-validation to balance performance and computation time.

Code Quality and Maintainability: Ensuring the pipeline followed best practices was challenging. Overcome by running Ruff for PEP-compliance and fixing linting issues.

2. Influence of Lab 4 Research

Lab 4 analysis guided feature selection, preprocessing choices, and model selection.

Decisions on which models to include (Logistic Regression, Random Forest, XGBoost) were based on Lab 4 evaluation.

Lab 4’s insights on scaling, encoding, and handling missing values directly informed preprocessing in Lab 5.

Feature importance and correlation analyses from Lab 4 helped determine which features to retain, ensuring model interpretability and performance.

3. Recommended Model for Deployment

XGBoost is recommended for deployment due to:

Highest ROC-AUC on the test set (0.951), showing strong discrimination ability.

Low Brier Score (0.052), indicating well-calibrated probability predictions.

SHAP analysis shows interpretable and business-relevant feature contributions.

Stable PSI values between train and test sets, suggesting reliable performance under deployment conditions.

