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

def get_linear_coefficients(model, feature_names):
    """Extract and sort coefficients from linear models."""
    if hasattr(model, 'coef_'):
        coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_})
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
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

def get_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
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


# =============================================================================
# Individual Model Training and Evaluation Functions
# =============================================================================

def train_and_evaluate_linear_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate Linear Regression model."""
    logger.info("=" * 60)
    logger.info("TRAINING: Linear Regression")
    logger.info("=" * 60)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Linear Regression',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_ridge(X_train, X_test, y_train, y_test, feature_names, alpha=1.0):
    """Train and evaluate Ridge Regression model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: Ridge Regression (alpha={alpha})")
    logger.info("=" * 60)
    
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Ridge Regression',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_lasso(X_train, X_test, y_train, y_test, feature_names, alpha=1.0):
    """Train and evaluate Lasso Regression model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: Lasso Regression (alpha={alpha})")
    logger.info("=" * 60)
    
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Lasso Regression',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test, feature_names, max_depth=10, min_samples_split=5, random_state=42):
    """Train and evaluate Decision Tree model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: Decision Tree (max_depth={max_depth})")
    logger.info("=" * 60)
    
    model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Decision Tree',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, feature_names, n_estimators=100, max_depth=10, random_state=42):
    """Train and evaluate Random Forest model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: Random Forest (n_estimators={n_estimators}, max_depth={max_depth})")
    logger.info("=" * 60)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Random Forest',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_gradient_boosting(X_train, X_test, y_train, y_test, feature_names, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """Train and evaluate Gradient Boosting model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: Gradient Boosting (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})")
    logger.info("=" * 60)
    
    model = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    
    logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
    logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
    logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
    
    results = {
        'Model': 'Gradient Boosting',
        'Train_R2': train_metrics['R2'],
        'Test_R2': test_metrics['R2'],
        'Train_RMSE': train_metrics['RMSE'],
        'Test_RMSE': test_metrics['RMSE'],
        'Train_MAE': train_metrics['MAE'],
        'Test_MAE': test_metrics['MAE'],
        'Test_MAPE': test_metrics['MAPE'],
        'CV_R2_Mean': cv_results['cv_mean'],
        'CV_R2_Std': cv_results['cv_std']
    }
    
    return model, y_pred_test, results


def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, feature_names, n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42):
    """Train and evaluate XGBoost model."""
    logger.info("=" * 60)
    logger.info(f"TRAINING: XGBoost (n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate})")
    logger.info("=" * 60)
    
    try:
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_metrics = evaluate_model(y_train, y_pred_train)
        test_metrics = evaluate_model(y_test, y_pred_test)
        cv_results = cross_validate_model(model, X_train, y_train, cv=5)
        
        logger.info(f"Training Complete - R2: {train_metrics['R2']:.4f}")
        logger.info(f"Test Results - R2: {test_metrics['R2']:.4f}, RMSE: {test_metrics['RMSE']:.2f}, MAE: {test_metrics['MAE']:.2f}")
        logger.info(f"Cross-Validation R2: {cv_results['cv_mean']:.4f} (+/- {cv_results['cv_std']:.4f})")
        
        results = {
            'Model': 'XGBoost',
            'Train_R2': train_metrics['R2'],
            'Test_R2': test_metrics['R2'],
            'Train_RMSE': train_metrics['RMSE'],
            'Test_RMSE': test_metrics['RMSE'],
            'Train_MAE': train_metrics['MAE'],
            'Test_MAE': test_metrics['MAE'],
            'Test_MAPE': test_metrics['MAPE'],
            'CV_R2_Mean': cv_results['cv_mean'],
            'CV_R2_Std': cv_results['cv_std']
        }
        
        return model, y_pred_test, results
    except ImportError:
        logger.warning("XGBoost not installed. Skipping XGBoost model.")
        return None, None, None


def compare_all_models(X_train, X_test, y_train, y_test, feature_names):
    """Train all models and return comparison results."""
    logger.info("=" * 60)
    logger.info("TRAINING ALL MODELS FOR COMPARISON")
    logger.info("=" * 60)
    
    all_results = []
    
    _, lr_pred, lr_results = train_and_evaluate_linear_regression(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(lr_results)
    
    _, ridge_pred, ridge_results = train_and_evaluate_ridge(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(ridge_results)
    
    _, lasso_pred, lasso_results = train_and_evaluate_lasso(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(lasso_results)
    
    _, dt_pred, dt_results = train_and_evaluate_decision_tree(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(dt_results)
    
    _, rf_pred, rf_results = train_and_evaluate_random_forest(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(rf_results)
    
    _, gb_pred, gb_results = train_and_evaluate_gradient_boosting(X_train, X_test, y_train, y_test, feature_names)
    all_results.append(gb_results)
    
    xgb_model, xgb_pred, xgb_results = train_and_evaluate_xgboost(X_train, X_test, y_train, y_test, feature_names)
    if xgb_results is not None:
        all_results.append(xgb_results)
    
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Test_R2', ascending=False).reset_index(drop=True)
    
    logger.info("=" * 60)
    logger.info("MODEL COMPARISON RESULTS (sorted by Test R2)")
    logger.info("=" * 60)
    logger.info(f"\n{results_df.to_string(index=False)}")
    
    return results_df
