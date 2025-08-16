# Lab-5-kamal-parekh

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
