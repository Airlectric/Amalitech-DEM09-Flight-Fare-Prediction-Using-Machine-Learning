"""Hyperparameter tuning and model optimization module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def tune_random_forest(X_train, y_train, X_test, y_test, n_iter=20, cv=3):
    """Use RandomizedSearchCV to tune RandomForest hyperparameters.
    
    Args:
        n_iter: Number of parameter settings sampled (default 20, reduce to 10-15 for faster tuning)
        cv: Number of cross-validation folds (default 3, use 5 for more robust results)
    
    Optimization notes:
    - Default (n_iter=20, cv=3): ~60 fits, takes 3-8 minutes
    - Fast (n_iter=10, cv=3): ~30 fits, takes 1-4 minutes
    - Thorough (n_iter=30, cv=5): ~150 fits, takes 15-30 minutes
    """
    logger.info(f"Starting Random Forest hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
    logger.info("Tip: Reduce n_iter to 10-15 if this takes too long")
    
    # Reduced parameter space for faster tuning
    param_dist = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=rf, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='r2', random_state=42, n_jobs=-1, verbose=1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    logger.info(f"Best Random Forest params: {best_params}")
    logger.info(f"Best CV R2 score: {random_search.best_score_:.4f}")
    y_pred = best_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    logger.info(f"Tuned Random Forest - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.2f}")
    return best_model, best_params, metrics, y_pred

def tune_gradient_boosting(X_train, y_train, X_test, y_test, n_iter=20, cv=3):
    """Tune GradientBoosting with RandomizedSearchCV.
    
    Args:
        n_iter: Number of parameter settings sampled (default 20, reduce to 10-15 for faster tuning)
        cv: Number of cross-validation folds (default 3, use 5 for more robust results)
    
    Optimization notes:
    - Default (n_iter=20, cv=3): ~60 fits, takes 5-10 minutes
    - Fast (n_iter=10, cv=3): ~30 fits, takes 2-5 minutes
    - Thorough (n_iter=30, cv=5): ~150 fits, takes 20-40 minutes
    """
    logger.info(f"Starting Gradient Boosting hyperparameter tuning (n_iter={n_iter}, cv={cv})...")
    logger.info("Tip: Reduce n_iter to 10-15 if this takes too long")
    
    # Reduced parameter space for faster tuning
    param_dist = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    gb = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        estimator=gb, param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring='r2', random_state=42, n_jobs=-1, verbose=1
    )
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    logger.info(f"Best Gradient Boosting params: {best_params}")
    logger.info(f"Best CV R2 score: {random_search.best_score_:.4f}")
    y_pred = best_model.predict(X_test)
    metrics = evaluate_metrics(y_test, y_pred)
    logger.info(f"Tuned Gradient Boosting - R2: {metrics['R2']:.4f}, RMSE: {metrics['RMSE']:.2f}")
    return best_model, best_params, metrics, y_pred

def compare_regularization(X_train, y_train, X_test, y_test, alphas=None, cv=3):
    """Compare Ridge and Lasso regularization to demonstrate effect on overfitting.

    Simplified implementation for faster execution - tests only 3 key alpha values
    plus a no-regularization baseline to show the regularization effect.

    Args:
        alphas: List of alpha values to test (default: [0.1, 1.0, 10.0])
        cv: Number of cross-validation folds (default 3)

    Optimization notes:
    - Fast mode ([0.1, 1.0, 10.0], cv=3): ~18 fits, takes 30-90 seconds
    - Original ([0.01, 0.1, 1.0, 10.0], cv=3): 24 fits, takes 2-5 minutes
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0]  # Simplified for faster execution

    logger.info(f"Comparing regularization effects (Ridge vs Lasso)...")
    logger.info(f"Testing alphas: {alphas} (simplified for speed)")
    logger.info(f"Total fits: {len(alphas) * 2} models")

    all_results = []

    # Add no-regularization baseline (alpha near 0)
    from sklearn.linear_model import LinearRegression
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)
    metrics_baseline = evaluate_metrics(y_test, y_pred_baseline)
    all_results.append({
        'alpha': 0.0, 'model': 'No Regularization',
        'R2_train': baseline.score(X_train, y_train),
        'R2_test': metrics_baseline['R2'],
        'RMSE': metrics_baseline['RMSE']
    })

    # Test Ridge and Lasso with selected alphas
    for alpha in alphas:
        for model_name, Model in [('Ridge', Ridge), ('Lasso', Lasso)]:
            model = Model(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_metrics(y_test, y_pred)
            all_results.append({
                'alpha': alpha, 'model': model_name,
                'R2_train': model.score(X_train, y_train),
                'R2_test': metrics['R2'],
                'RMSE': metrics['RMSE']
            })

    results_df = pd.DataFrame(all_results)
    logger.info(f"\nRegularization comparison:\n{results_df.to_string(index=False)}")
    return results_df

def plot_regularization_effect(reg_results):
    """Plot train vs test R2 to demonstrate how Ridge and Lasso affect overfitting.

    Shows bias-variance tradeoff: as alpha increases, training performance decreases
    but test performance may improve (reduced overfitting).
    """
    plt.figure(figsize=(12, 5))

    # Plot 1: Train vs Test R2 for all models
    plt.subplot(1, 2, 1)
    for model_name in ['No Regularization', 'Ridge', 'Lasso']:
        model_data = reg_results[reg_results['model'] == model_name]
        if len(model_data) > 0:
            alphas = model_data['alpha'].values
            # Use numeric x-positions for better spacing
            x_pos = range(len(alphas))
            plt.plot(x_pos, model_data['R2_train'].values, marker='o',
                    label=f'{model_name} (Train)', linestyle='--', alpha=0.7)
            plt.plot(x_pos, model_data['R2_test'].values, marker='s',
                    label=f'{model_name} (Test)', linewidth=2)

    plt.xlabel('Alpha Value')
    plt.ylabel('R² Score')
    plt.title('Regularization Effect: Train vs Test Performance')
    plt.xticks(range(len(reg_results['alpha'].unique())),
               [f'{a:.1f}' for a in sorted(reg_results['alpha'].unique())])
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)

    # Plot 2: Overfitting gap (Train R2 - Test R2)
    plt.subplot(1, 2, 2)
    for model_name in ['No Regularization', 'Ridge', 'Lasso']:
        model_data = reg_results[reg_results['model'] == model_name]
        if len(model_data) > 0:
            alphas = model_data['alpha'].values
            gap = model_data['R2_train'].values - model_data['R2_test'].values
            x_pos = range(len(alphas))
            plt.plot(x_pos, gap, marker='o', label=model_name, linewidth=2)

    plt.xlabel('Alpha Value')
    plt.ylabel('Overfitting Gap (Train R² - Test R²)')
    plt.title('How Regularization Reduces Overfitting')
    plt.xticks(range(len(reg_results['alpha'].unique())),
               [f'{a:.1f}' for a in sorted(reg_results['alpha'].unique())])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No gap')

    plt.tight_layout()
    plt.show()
    logger.info("Plotted regularization effect: train/test performance and overfitting gap")

def cross_validate_best_model(model, X_train, y_train, cv=3):
    """Run cross-validation on the best model.
    
    Args:
        cv: Number of cross-validation folds (default 3, use 5 for more robust results)
    
    Optimization notes:
    - cv=3: 3 fits, takes 30 seconds - 2 minutes
    - cv=5: 5 fits, takes 1-3 minutes
    - cv=10: 10 fits, takes 2-6 minutes
    """
    logger.info(f"Performing {cv}-fold cross-validation on best model...")
    logger.info("Tip: Use cv=5 for publication-quality results, cv=3 for faster iteration")
    
    r2_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
    mae_scores = -cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_absolute_error')
    rmse_scores = np.sqrt(-cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error'))
    results = {
        'cv_folds': cv, 'R2_mean': r2_scores.mean(), 'R2_std': r2_scores.std(),
        'MAE_mean': mae_scores.mean(), 'MAE_std': mae_scores.std(),
        'RMSE_mean': rmse_scores.mean(), 'RMSE_std': rmse_scores.std(),
        'R2_scores': r2_scores, 'MAE_scores': mae_scores, 'RMSE_scores': rmse_scores
    }
    logger.info(f"Cross-validation results: R2: {results['R2_mean']:.4f} (+/- {results['R2_std']:.4f})")
    return results

# Convenience functions for quick tuning
def tune_random_forest_quick(X_train, y_train, X_test, y_test):
    """Quick Random Forest tuning (n_iter=10, cv=3) - ~1-4 minutes."""
    return tune_random_forest(X_train, y_train, X_test, y_test, n_iter=10, cv=3)

def tune_gradient_boosting_quick(X_train, y_train, X_test, y_test):
    """Quick Gradient Boosting tuning (n_iter=10, cv=3) - ~2-5 minutes."""
    return tune_gradient_boosting(X_train, y_train, X_test, y_test, n_iter=10, cv=3)

def tune_random_forest_thorough(X_train, y_train, X_test, y_test):
    """Thorough Random Forest tuning (n_iter=30, cv=5) - ~15-30 minutes."""
    return tune_random_forest(X_train, y_train, X_test, y_test, n_iter=30, cv=5)

def tune_gradient_boosting_thorough(X_train, y_train, X_test, y_test):
    """Thorough Gradient Boosting tuning (n_iter=30, cv=5) - ~20-40 minutes."""
    return tune_gradient_boosting(X_train, y_train, X_test, y_test, n_iter=30, cv=5)
