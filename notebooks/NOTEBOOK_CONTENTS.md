# Flight Fare Prediction - Notebook Cell Contents

Copy each section below into a Jupyter notebook cell.

**Total Cells: 75** (including model saving cell for deployment)

---

## CELL 1 - Markdown

```markdown
# Flight Fare Prediction Using Machine Learning
## AmaliTech NSS - Machine Learning Module Lab

**Objective:** Build an end-to-end ML pipeline for predicting flight fares.

**ML Task:** Supervised Regression
**Target Variable:** Total Fare (BDT)
```

---

## CELL 2 - Code

```python
import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
```

---

## CELL 3 - Markdown

```markdown
## Step 1: Problem Definition & Data Understanding
```

---

## CELL 4 - Code

```python
from src.data_loader import load_dataset, inspect_dataset

df = load_dataset()
summary = inspect_dataset(df)
df.head(10)
```

---

## CELL 5 - Code

```python
df.info()
```

---

## CELL 6 - Code

```python
df.describe()
```

---

## CELL 7 - Code

```python
print('Missing values per column:')
print(df.isnull().sum())
print(f'\nTotal duplicate rows: {df.duplicated().sum()}')
```

---

## CELL 8 - Markdown

```markdown
### Initial Observations
- **Dataset shape:** See `df.shape` after execution
- **Target variable:** Total Fare (BDT)
- **Missing data:** See above
- **Duplicates:** See above
```

---

## CELL 9 - Markdown

```markdown
## Step 2: Data Cleaning & Preprocessing
```

---

## CELL 10 - Code

```python
from src.data_cleaning import clean_dataset

df_clean = clean_dataset(df)
print(f'Shape before cleaning: {df.shape}')
print(f'Shape after cleaning:  {df_clean.shape}')
df_clean.head()
```

---

## CELL 11 - Code

```python
print('Missing values after cleaning:')
print(df_clean.isnull().sum())
print(f'\nDuplicate rows: {df_clean.duplicated().sum()}')
print(f'\nData types:\n{df_clean.dtypes}')
```

---

## CELL 12 - Markdown

```markdown
## Step 3: Exploratory Data Analysis (EDA)
```

---

## CELL 13 - Markdown

```markdown
### 3.1 Descriptive Statistics
```

---

## CELL 14 - Code

```python
from src.eda import summarize_fares, compute_kpis

fare_summary = summarize_fares(df_clean)
fare_summary
```

---

## CELL 15 - Code

```python
kpis = compute_kpis(df_clean)
for key, value in kpis.items():
    print(f'{key}: {value}')
```

---

## CELL 16 - Markdown

```markdown
### 3.2 Visual Analysis
```

---

## CELL 17 - Code

```python
from src.eda import plot_fare_distribution
plot_fare_distribution(df_clean)
```

---

## CELL 18 - Code

```python
from src.eda import plot_fare_by_airline
plot_fare_by_airline(df_clean)
```

---

## CELL 19 - Code

```python
from src.eda import plot_fare_boxplots
plot_fare_boxplots(df_clean)
```

---

## CELL 20 - Code

```python
from src.eda import plot_fare_by_season
plot_fare_by_season(df_clean)
```

---

## CELL 21 - Code

```python
from src.eda import plot_correlation_heatmap
plot_correlation_heatmap(df_clean)
```

---

## CELL 22 - Markdown

```markdown
### 3.3 Key EDA Findings
```

---

## CELL 23 - Markdown

```markdown
## Step 4: Feature Engineering
```

---

## CELL 24 - Code

```python
from src.feature_engineering import run_feature_pipeline

X_train, X_test, y_train, y_test, scaler, feature_names = run_feature_pipeline(df_clean)

print(f'Training set: {X_train.shape}')
print(f'Test set:     {X_test.shape}')
print(f'Features:     {len(feature_names)}')
print(f'\nFeature names: {feature_names}')
```

---

## CELL 25 - Code

```python
print('Target variable (y_train) statistics:')
print(y_train.describe())
print(f'\nNaN values in X_train: {X_train.isnull().sum().sum()}')
print(f'NaN values in X_test: {X_test.isnull().sum().sum()}')
```

---

## CELL 26 - Markdown

```markdown
## Step 5: Baseline Models
```

---

## CELL 27 - Markdown

```markdown
### 5.1 Linear Regression
```

---

## CELL 28 - Code

```python
from src.models import (
    train_and_evaluate_linear_regression,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_linear_coefficients
)

lr_model, lr_pred, lr_results = train_and_evaluate_linear_regression(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('LINEAR REGRESSION RESULTS')
print('='*60)
print(f'Train R2:  {lr_results["Train_R2"]:.6f}')
print(f'Test R2:   {lr_results["Test_R2"]:.6f}')
print(f'Test RMSE: {lr_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {lr_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {lr_results["CV_R2_Mean"]:.6f} (+/- {lr_results["CV_R2_Std"]:.6f})')
```

---

## CELL 29 - Code

```python
plot_actual_vs_predicted(y_test.values, lr_pred, 'Linear Regression')
```

---

## CELL 30 - Code

```python
plot_residuals(y_test.values, lr_pred, 'Linear Regression')
```

---

## CELL 31 - Code

```python
plot_linear_coefficients(lr_model, feature_names, 'Linear Regression')
```

---

## CELL 32 - Markdown

```markdown
### 5.2 Ridge Regression
```

---

## CELL 33 - Code

```python
from src.models import train_and_evaluate_ridge

ridge_model, ridge_pred, ridge_results = train_and_evaluate_ridge(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('RIDGE REGRESSION RESULTS')
print('='*60)
print(f'Train R2:  {ridge_results["Train_R2"]:.6f}')
print(f'Test R2:   {ridge_results["Test_R2"]:.6f}')
print(f'Test RMSE: {ridge_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {ridge_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {ridge_results["CV_R2_Mean"]:.6f} (+/- {ridge_results["CV_R2_Std"]:.6f})')
```

---

## CELL 34 - Code

```python
plot_actual_vs_predicted(y_test.values, ridge_pred, 'Ridge Regression')
plot_residuals(y_test.values, ridge_pred, 'Ridge Regression')
```

---

## CELL 35 - Markdown

```markdown
### 5.3 Lasso Regression
```

---

## CELL 36 - Code

```python
from src.models import train_and_evaluate_lasso

lasso_model, lasso_pred, lasso_results = train_and_evaluate_lasso(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('LASSO REGRESSION RESULTS')
print('='*60)
print(f'Train R2:  {lasso_results["Train_R2"]:.6f}')
print(f'Test R2:   {lasso_results["Test_R2"]:.6f}')
print(f'Test RMSE: {lasso_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {lasso_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {lasso_results["CV_R2_Mean"]:.6f} (+/- {lasso_results["CV_R2_Std"]:.6f})')
```

---

## CELL 37 - Code

```python
plot_actual_vs_predicted(y_test.values, lasso_pred, 'Lasso Regression')
plot_residuals(y_test.values, lasso_pred, 'Lasso Regression')
```

---

## CELL 38 - Markdown

```markdown
## Step 6: Advanced Models
```

---

## CELL 39 - Markdown

```markdown
### 6.1 Decision Tree Regressor
```

---

## CELL 40 - Code

```python
from src.models import (
    train_and_evaluate_decision_tree,
    plot_feature_importance
)

dt_model, dt_pred, dt_results = train_and_evaluate_decision_tree(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('DECISION TREE RESULTS')
print('='*60)
print(f'Train R2:  {dt_results["Train_R2"]:.6f}')
print(f'Test R2:   {dt_results["Test_R2"]:.6f}')
print(f'Test RMSE: {dt_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {dt_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {dt_results["CV_R2_Mean"]:.6f} (+/- {dt_results["CV_R2_Std"]:.6f})')
```

---

## CELL 41 - Code

```python
plot_actual_vs_predicted(y_test.values, dt_pred, 'Decision Tree')
plot_residuals(y_test.values, dt_pred, 'Decision Tree')
plot_feature_importance(dt_model, feature_names, 'Decision Tree')
```

---

## CELL 42 - Markdown

```markdown
### 6.2 Random Forest Regressor
```

---

## CELL 43 - Code

```python
from src.models import train_and_evaluate_random_forest

rf_model, rf_pred, rf_results = train_and_evaluate_random_forest(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('RANDOM FOREST RESULTS')
print('='*60)
print(f'Train R2:  {rf_results["Train_R2"]:.6f}')
print(f'Test R2:   {rf_results["Test_R2"]:.6f}')
print(f'Test RMSE: {rf_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {rf_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {rf_results["CV_R2_Mean"]:.6f} (+/- {rf_results["CV_R2_Std"]:.6f})')
```

---

## CELL 44 - Code

```python
plot_actual_vs_predicted(y_test.values, rf_pred, 'Random Forest')
plot_residuals(y_test.values, rf_pred, 'Random Forest')
plot_feature_importance(rf_model, feature_names, 'Random Forest')
```

---

## CELL 45 - Markdown

```markdown
### 6.3 Gradient Boosting Regressor
```

---

## CELL 46 - Code

```python
from src.models import train_and_evaluate_gradient_boosting

gb_model, gb_pred, gb_results = train_and_evaluate_gradient_boosting(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('GRADIENT BOOSTING RESULTS')
print('='*60)
print(f'Train R2:  {gb_results["Train_R2"]:.6f}')
print(f'Test R2:   {gb_results["Test_R2"]:.6f}')
print(f'Test RMSE: {gb_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {gb_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {gb_results["CV_R2_Mean"]:.6f} (+/- {gb_results["CV_R2_Std"]:.6f})')
```

---

## CELL 47 - Code

```python
plot_actual_vs_predicted(y_test.values, gb_pred, 'Gradient Boosting')
plot_residuals(y_test.values, gb_pred, 'Gradient Boosting')
plot_feature_importance(gb_model, feature_names, 'Gradient Boosting')
```

---

## CELL 48 - Markdown

```markdown
### 6.4 XGBoost Regressor
```

---

## CELL 49 - Code

```python
from src.models import train_and_evaluate_xgboost

xgb_model, xgb_pred, xgb_results = train_and_evaluate_xgboost(
    X_train, X_test, y_train, y_test, feature_names
)

print('\n' + '='*60)
print('XGBOOST RESULTS')
print('='*60)
print(f'Train R2:  {xgb_results["Train_R2"]:.6f}')
print(f'Test R2:   {xgb_results["Test_R2"]:.6f}')
print(f'Test RMSE: {xgb_results["Test_RMSE"]:.2f} BDT')
print(f'Test MAE:  {xgb_results["Test_MAE"]:.2f} BDT')
print(f'CV R2:     {xgb_results["CV_R2_Mean"]:.6f} (+/- {xgb_results["CV_R2_Std"]:.6f})')
```

---

## CELL 50 - Code

```python
plot_actual_vs_predicted(y_test.values, xgb_pred, 'XGBoost')
plot_residuals(y_test.values, xgb_pred, 'XGBoost')
plot_feature_importance(xgb_model, feature_names, 'XGBoost')
```

---

## CELL 51 - Markdown

```markdown
### 6.5 Model Comparison Summary
```

---

## CELL 52 - Code

```python
import pandas as pd

all_results = [lr_results, ridge_results, lasso_results, dt_results, rf_results, gb_results, xgb_results]
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)

print('\n' + '='*80)
print('MODEL COMPARISON (sorted by Test R2)')
print('='*80)
print(results_df.to_string(index=False))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

r = results_df.sort_values('Test_R2', ascending=True)
r.plot.barh(x='Model', y='Test_R2', ax=axes[0], legend=False, color='steelblue')
axes[0].set_title('R2 Score (higher is better)')
axes[0].set_xlabel('R2 Score')

r = results_df.sort_values('Test_RMSE', ascending=False)
r.plot.barh(x='Model', y='Test_RMSE', ax=axes[1], legend=False, color='orange')
axes[1].set_title('RMSE (lower is better)')
axes[1].set_xlabel('RMSE (BDT)')

r = results_df.sort_values('Test_MAE', ascending=False)
r.plot.barh(x='Model', y='Test_MAE', ax=axes[2], legend=False, color='green')
axes[2].set_title('MAE (lower is better)')
axes[2].set_xlabel('MAE (BDT)')

plt.tight_layout()
plt.show()
```

---

## CELL 53 - Markdown

```markdown
### 6.6 Hyperparameter Tuning - Random Forest
```

---

## CELL 54 - Code

```python
from src.optimization import tune_random_forest

print('\n' + '='*60)
print('TUNING RANDOM FOREST')
print('='*60)

# Optimized defaults: n_iter=20, cv=3 (~3-8 minutes)
# For faster tuning: use tune_random_forest_quick() with n_iter=10 (~1-4 minutes)
# For thorough tuning: use tune_random_forest_thorough() with n_iter=30, cv=5 (~15-30 minutes)

rf_tuned, rf_params, rf_metrics, rf_pred_tuned = tune_random_forest(
    X_train, y_train, X_test, y_test
)

print(f'\nBest RF Parameters: {rf_params}')
print(f'Tuned RF - R2: {rf_metrics["R2"]:.6f}, RMSE: {rf_metrics["RMSE"]:.2f}, MAE: {rf_metrics["MAE"]:.2f}')
```

---

## CELL 55 - Code

```python
plot_actual_vs_predicted(y_test.values, rf_pred_tuned, 'Random Forest (Tuned)')
plot_residuals(y_test.values, rf_pred_tuned, 'Random Forest (Tuned)')
plot_feature_importance(rf_tuned, feature_names, 'Random Forest (Tuned)')
```

---

## CELL 56 - Markdown

```markdown
### 6.7 Hyperparameter Tuning - Gradient Boosting
```

---

## CELL 57 - Code

```python
from src.optimization import tune_gradient_boosting

print('\n' + '='*60)
print('TUNING GRADIENT BOOSTING')
print('='*60)

# Optimized defaults: n_iter=20, cv=3 (~5-10 minutes)
# For faster tuning: use tune_gradient_boosting_quick() with n_iter=10 (~2-5 minutes)
# For thorough tuning: use tune_gradient_boosting_thorough() with n_iter=30, cv=5 (~20-40 minutes)

gb_tuned, gb_params, gb_metrics, gb_pred_tuned = tune_gradient_boosting(
    X_train, y_train, X_test, y_test
)

print(f'\nBest GB Parameters: {gb_params}')
print(f'Tuned GB - R2: {gb_metrics["R2"]:.6f}, RMSE: {gb_metrics["RMSE"]:.2f}, MAE: {gb_metrics["MAE"]:.2f}')
```

---

## CELL 58 - Code

```python
plot_actual_vs_predicted(y_test.values, gb_pred_tuned, 'Gradient Boosting (Tuned)')
plot_residuals(y_test.values, gb_pred_tuned, 'Gradient Boosting (Tuned)')
plot_feature_importance(gb_tuned, feature_names, 'Gradient Boosting (Tuned)')
```

---

## CELL 59 - Markdown

```markdown
### 6.8 Regularization Analysis
```

---

## CELL 60 - Code

```python
from src.optimization import compare_regularization, plot_regularization_effect

print('\n' + '='*60)
print('REGULARIZATION ANALYSIS')
print('='*60)

# NOTE: This cell tests multiple alpha values with cross-validation.
# If it takes too long (>10 minutes), use these optimized parameters:
#   alphas=[0.01, 0.1, 1.0, 10.0]  - fewer values
#   cv=3                            - fewer folds (faster)

# Option A: Full analysis (may take 1-2 hours with large datasets)
# reg_results = compare_regularization(X_train, y_train, X_test, y_test)

# Option B: Quick analysis (recommended - takes 2-5 minutes)
reg_results = compare_regularization(
    X_train, y_train, X_test, y_test,
    alphas=[0.01, 0.1, 1.0, 10.0],  # Reduced from 6 to 4 values
    cv=3                             # Reduced from 5 to 3 folds
)

plot_regularization_effect(reg_results)
reg_results
```

---

## CELL 61 - Markdown

```markdown
### 6.9 Cross-Validation of Best Model
```

---

## CELL 62 - Code

```python
from src.optimization import cross_validate_best_model

if rf_metrics['R2'] >= gb_metrics['R2']:
    best_tuned_model = rf_tuned
    best_tuned_name = 'Random Forest (Tuned)'
    best_tuned_pred = rf_pred_tuned
    best_tuned_metrics = rf_metrics
else:
    best_tuned_model = gb_tuned
    best_tuned_name = 'Gradient Boosting (Tuned)'
    best_tuned_pred = gb_pred_tuned
    best_tuned_metrics = gb_metrics

print('\n' + '='*60)
print('CROSS-VALIDATION OF BEST MODEL')
print('='*60)
print(f'Best Tuned Model: {best_tuned_name}')

# Default cv=3 (~30 seconds - 2 minutes)
# Use cv=5 for more robust results (~1-3 minutes)
cv_results = cross_validate_best_model(best_tuned_model, X_train, y_train)

print(f'CV R2:  {cv_results["R2_mean"]:.6f} (+/- {cv_results["R2_std"]:.6f})')
```

---

## CELL 63 - Markdown

```markdown
### 6.10 Final Model Comparison
```

---

## CELL 64 - Code

```python
import os

final_results = results_df.copy()

tuned_rf = pd.DataFrame([{
    'Model': 'Random Forest (Tuned)',
    'Train_R2': rf_metrics['R2'],
    'Test_R2': rf_metrics['R2'],
    'Train_RMSE': rf_metrics['RMSE'],
    'Test_RMSE': rf_metrics['RMSE'],
    'Train_MAE': rf_metrics['MAE'],
    'Test_MAE': rf_metrics['MAE'],
    'CV_R2_Mean': cv_results['R2_mean'],
    'CV_R2_Std': cv_results['R2_std']
}])

tuned_gb = pd.DataFrame([{
    'Model': 'Gradient Boosting (Tuned)',
    'Train_R2': gb_metrics['R2'],
    'Test_R2': gb_metrics['R2'],
    'Train_RMSE': gb_metrics['RMSE'],
    'Test_RMSE': gb_metrics['RMSE'],
    'Train_MAE': gb_metrics['MAE'],
    'Test_MAE': gb_metrics['MAE'],
    'CV_R2_Mean': cv_results['R2_mean'],
    'CV_R2_Std': cv_results['R2_std']
}])

final_results = pd.concat([final_results, tuned_rf, tuned_gb], ignore_index=True)
final_results = final_results.sort_values('Test_R2', ascending=False).reset_index(drop=True)

os.makedirs('reports', exist_ok=True)
final_results.to_csv('reports/model_comparison.csv', index=False)

print('\n' + '='*80)
print('FINAL MODEL COMPARISON (saved to reports/model_comparison.csv)')
print('='*80)
print(final_results.to_string(index=False))
```

---

## CELL 65 - Code

```python
# Save Best Model and Scaler for Deployment
import os
import pickle

# Create models directory
os.makedirs('../models', exist_ok=True)

# Save the best tuned model
best_model_path = '../models/best_model.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(best_tuned_model, f)
print(f'Best model saved to: {best_model_path}')

# Save the scaler
scaler_path = '../models/scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f'Scaler saved to: {scaler_path}')

# Save feature names for reference
feature_names_path = '../models/feature_names.txt'
with open(feature_names_path, 'w') as f:
    f.write('\n'.join(feature_names))
print(f'Feature names saved to: {feature_names_path}')

print('\n' + '='*80)
print('MODEL ARTIFACTS SAVED SUCCESSFULLY')
print('='*80)
print(f'  - Model type: {type(best_tuned_model).__name__}')
print(f'  - Model size: {os.path.getsize(best_model_path) / 1024:.2f} KB')
print(f'  - Scaler type: {type(scaler).__name__}')
print(f'  - Number of features: {len(feature_names)}')
print(f'\nThese files can be used for deployment in Streamlit, Flask, or Docker.')
```

---

## CELL 66 - Markdown

```markdown
## Step 7: Model Interpretation & Insights
```

---

## CELL 67 - Markdown

```markdown
### 7.1 Feature Importance Analysis
```

---

## CELL 68 - Code

```python
from src.interpretation import plot_linear_coefficients, plot_tree_feature_importance

plot_linear_coefficients(lr_model, feature_names, 'Linear Regression')
```

---

## CELL 69 - Code

```python
plot_tree_feature_importance(rf_tuned, feature_names, 'Random Forest (Tuned)')
```

---

## CELL 70 - Code

```python
from src.interpretation import plot_best_model_predictions

plot_best_model_predictions(y_test.values, best_tuned_pred, best_tuned_name)
```

---

## CELL 71 - Markdown

```markdown
### 7.2 Key Insights & Recommendations
```

---

## CELL 72 - Code

```python
from src.interpretation import generate_insights, summarize_findings
from src.eda import summarize_fares

fare_summary = summarize_fares(df_clean)
insights = generate_insights(df_clean, best_tuned_model, feature_names, final_results, fare_summary)
print(summarize_findings(insights))
```

---

## CELL 73 - Markdown

```markdown
## Conclusion
```

---

## CELL 74 - Markdown

```markdown
### Summary
Built an end-to-end ML pipeline for predicting flight fares from the Bangladesh Flight Price Dataset.

### Key Results
- Best performing model: See `final_results`
- All metrics saved to `reports/model_comparison.csv`

### Recommendations
- Use the tuned model for production deployment
- Feature importance reveals key fare drivers
- Consider seasonal patterns for pricing strategies

### Limitations
- External factors (fuel prices, events) not captured
- Model may vary for underrepresented routes

### Future Work
- Incorporate external data sources
- Experiment with ensemble methods
- Deploy as REST API for real-time predictions
```

---

## CELL 75 - Code

```python
# End of notebook
```
