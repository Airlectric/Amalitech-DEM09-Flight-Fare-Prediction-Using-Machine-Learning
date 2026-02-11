"""Model interpretation and insights generation module."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.logger import get_logger

logger = get_logger(__name__)

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

def get_tree_feature_importance(model, feature_names):
    """Extract feature importance from tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        logger.info(f"Extracted feature importances from {model.__class__.__name__}")
        return importance_df
    else:
        logger.warning(f"Model {model.__class__.__name__} does not have feature importances")
        return None

def plot_tree_feature_importance(model, feature_names, model_name="Model", top_n=20):
    """Plot top N feature importances for tree-based models."""
    importance_df = get_tree_feature_importance(model, feature_names)
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

def plot_best_model_predictions(y_true, y_pred, model_name="Model"):
    """Final actual vs predicted scatter for the best model."""
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.5, edgecolors='k', linewidths=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal (y=x)')
    plt.xlabel('Actual Total Fare (BDT)')
    plt.ylabel('Predicted Total Fare (BDT)')
    plt.title(f'{model_name}: Best Model Predictions')
    plt.legend()
    plt.tight_layout()
    plt.show()
    logger.info(f"Plotted best model predictions for {model_name}")

def generate_insights(df, best_model, feature_names, results_df, fare_summary=None):
    """Generate and log key insights about flight fares."""
    logger.info("Generating business insights...")
    insights = {}

    if hasattr(best_model, 'feature_importances_'):
        importance_df = get_tree_feature_importance(best_model, feature_names)
        if importance_df is not None:
            top_5 = importance_df.head(5)
            insights['Top 5 Most Important Features'] = top_5.to_dict('records')
            logger.info(f"Top 5 features: {top_5.to_dict('records')}")
    elif hasattr(best_model, 'coef_'):
        coef_df = get_linear_coefficients(best_model, feature_names)
        if coef_df is not None:
            top_5 = coef_df.head(5)
            insights['Top 5 Most Important Features'] = top_5.to_dict('records')
            logger.info(f"Top 5 features: {top_5.to_dict('records')}")

    if fare_summary is not None:
        most_expensive = fare_summary.head(3)
        cheapest = fare_summary.tail(3)
        insights['Airline Pricing Patterns'] = {
            'Most Expensive': most_expensive.index.tolist(),
            'Cheapest': cheapest.index.tolist()
        }

    if 'Seasonality' in df.columns:
        seasonal_avg = df.groupby('Seasonality')['Total Fare (BDT)'].mean().sort_values(ascending=False)
        insights['Seasonal Patterns'] = seasonal_avg.to_dict()

    if 'Class' in df.columns:
        class_avg = df.groupby('Class')['Total Fare (BDT)'].mean().sort_values(ascending=False)
        insights['Class Pricing'] = class_avg.to_dict()

    if results_df is not None and len(results_df) > 0:
        best_idx = results_df['Test_R2'].idxmax()
        best_row = results_df.loc[best_idx]
        insights['Best Model Recommendation'] = {
            'Model': best_row['Model'],
            'Test R2': f"{best_row['Test_R2']:.4f}",
            'Test RMSE': f"{best_row['Test_RMSE']:.2f}",
            'CV R2 Mean': f"{best_row['CV_R2_Mean']:.4f}"
        }

    for key, value in insights.items():
        logger.info(f"\n--- {key} ---")
        logger.info(str(value))

    return insights

def summarize_findings(insights):
    """Generate a summary report string."""
    report = []
    report.append("=" * 60)
    report.append("FLIGHT FARE PREDICTION - ANALYSIS SUMMARY")
    report.append("=" * 60)

    if 'Top 5 Most Important Features' in insights:
        report.append("\n1. KEY FACTORS AFFECTING FLIGHT FARES:")
        for i, f in enumerate(insights['Top 5 Most Important Features'], 1):
            report.append(f"   {i}. {f}")

    if 'Airline Pricing Patterns' in insights:
        patterns = insights['Airline Pricing Patterns']
        report.append(f"\n2. AIRLINE PRICING PATTERNS:")
        report.append(f"   Most Expensive: {', '.join(patterns['Most Expensive'])}")
        report.append(f"   Cheapest: {', '.join(patterns['Cheapest'])}")

    if 'Seasonal Patterns' in insights:
        report.append("\n3. SEASONAL PRICE VARIATION:")
        for season, avg_fare in insights['Seasonal Patterns'].items():
            report.append(f"   {season}: {avg_fare:,.2f} BDT")

    if 'Best Model Recommendation' in insights:
        rec = insights['Best Model Recommendation']
        report.append("\n4. MODEL RECOMMENDATION:")
        report.append(f"   Best Model: {rec['Model']}")
        report.append(f"   Test R2: {rec['Test R2']}")
        report.append(f"   Test RMSE: {rec['Test RMSE']} BDT")

    report.append("\n" + "=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)
    return "\n".join(report)
