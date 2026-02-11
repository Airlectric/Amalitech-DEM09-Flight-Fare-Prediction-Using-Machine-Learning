"""Model training and evaluation module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(y_true, y_pred):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2, 'MAPE': mape}

def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    logger.info("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    logger.info("Linear Regression training completed")
    return model

def train_ridge_regression(X_train, y_train, alpha=1.0):
    """Train a Ridge Regression model."""
    logger.info(f"Training Ridge Regression model (alpha={alpha})...")
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    logger.info("Ridge Regression training completed")
    return model

def train_lasso_regression(X_train, y_train, alpha=1.0):
    """Train a Lasso Regression model."""
    logger.info(f"Training Lasso Regression model (alpha={alpha})...")
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    logger.info("Lasso Regression training completed")
    return model

def train_decision_tree(X_train, y_train, max_depth=10, min_samples_split=5, random_state=42):
    """Train a Decision Tree Regressor."""
    logger.info(f"Training Decision Tree model (max_depth={max_depth})...")
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Decision Tree training completed")
    return model

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42):
    """Train a Random Forest Regressor."""
    logger.info(f"Training Random Forest model (n_estimators={n_estimators}, max_depth={max_depth})...")
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info("Random Forest training completed")
    return model

def train_gradient_boosting(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """Train a Gradient Boosting Regressor."""
    logger.info(f"Training Gradient Boosting model (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting training completed")
    return model

def train_xgboost(X_train, y_train, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """Train an XGBoost Regressor."""
    try:
        from xgboost import XGBRegressor
        logger.info(f"Training XGBoost model (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})...")
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        logger.info("XGBoost training completed")
        return model
    except ImportError:
        logger.warning("XGBoost not installed. Skipping XGBoost model.")
        return None

def cross_validate_model(model, X_train, y_train, cv=5, scoring='r2'):
    """Perform cross-validation on a model."""
    logger.info(f"Performing {cv}-fold cross-validation...")
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    return {'cv_mean': scores.mean(), 'cv_std': scores.std(), 'cv_scores': scores}

def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """Scatter plot of actual vs predicted with ideal line."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    plt.xlabel('Actual Total Fare (BDT)')
    plt.ylabel('Predicted Total Fare (BDT)')
    plt.title(f'{model_name}: Actual vs Predicted')
    plt.legend()
    plt.tight_layout()
    plt.show()
    logger.info(f"Plotted actual vs predicted for {model_name}")

def plot_residuals(y_true, y_pred, model_name="Model"):
    """Residual plot (predicted vs residuals) to check for patterns."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Total Fare (BDT)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'{model_name}: Residual Analysis')
    plt.tight_layout()
    plt.show()
    logger.info(f"Plotted residuals for {model_name}")

def get_linear_coefficients(model, feature_names):
    """Extract and sort coefficients from linear models."""
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        logger.info(f"Extracted {len(coef_df)} coefficients from {model.__class__.__name__}")
        return coef_df
    else:
        logger.warning(f"Model {model.__class__.__name__} does not have coefficients")
        return None

def plot_linear_coefficients(model, feature_names, model_name="Model", top_n=20):
    """Plot top N coefficients (absolute value) for linear models."""
    coef_df = get_linear_coefficients(model, feature_names)
    if coef_df is None:
        return
    plt.figure(figsize=(12, 8))
    top_coef = coef_df.head(top_n)
    colors = ['green' if c > 0 else 'red' for c in top_coef['Coefficient']]
    plt.barh(range(len(top_coef)), top_coef['Coefficient'].values, color=colors)
    plt.yticks(range(len(top_coef)), top_coef['Feature'].values)
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title(f'{model_name}: Top {top_n} Feature Coefficients')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    logger.info(f"Plotted top {top_n} coefficients for {model_name}")

def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        logger.info(f"Extracted feature importances from {model.__class__.__name__}")
        return importance_df
    else:
        logger.warning(f"Model {model.__class__.__name__} does not have feature importances")
        return None

def plot_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """Plot top N feature importances for tree-based models."""
    importance_df = get_feature_importance(model, feature_names)
    if importance_df is None:
        return
    plt.figure(figsize=(12, 8))
    top_importance = importance_df.head(top_n)
    plt.barh(range(len(top_importance)), top_importance['Importance'].values)
    plt.yticks(range(len(top_importance)), top_importance['Feature'].values)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'{model_name}: Top {top_n} Feature Importances')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    logger.info(f"Plotted top {top_n} feature importances for {model_name}")

def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple baseline models and return evaluation results."""
    models = {
        'Linear Regression': train_linear_regression,
        'Ridge Regression': train_ridge_regression,
        'Lasso Regression': train_lasso_regression,
        'Decision Tree': train_decision_tree,
        'Random Forest': train_random_forest,
        'Gradient Boosting': train_gradient_boosting,
        'XGBoost': train_xgboost
    }
    results = []
    for name, train_func in models.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*50}")
        model = train_func(X_train, y_train)
        if model is None:
            continue
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)
        cv_results = cross_validate_model(model, X_train, y_train, cv=5, scoring='r2')
        logger.info(f"{name} - Test R2: {test_metrics['R2']:.4f}, Test RMSE: {test_metrics['RMSE']:.2f}")
        results.append({
            'Model': name,
            'Train_R2': train_metrics['R2'],
            'Test_R2': test_metrics['R2'],
            'Train_RMSE': train_metrics['RMSE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Train_MAE': train_metrics['MAE'],
            'Test_MAE': test_metrics['MAE'],
            'Test_MAPE': test_metrics['MAPE'],
            'CV_R2_Mean': cv_results['cv_mean'],
            'CV_R2_Std': cv_results['cv_std']
        })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False)
    logger.info(f"\n{'='*50}")
    logger.info("Model Comparison Results (sorted by Test R2)")
    logger.info(f"{'='*50}")
    logger.info(f"\n{results_df.to_string(index=False)}")
    return results_df
