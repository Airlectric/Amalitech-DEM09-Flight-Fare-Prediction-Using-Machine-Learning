"""Exploratory Data Analysis module."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import get_logger

logger = get_logger(__name__)

def get_price_column(df: pd.DataFrame) -> str:
    """Detect the price/fare column in the dataset."""
    price_candidates = ['Total Fare (BDT)', 'Base Fare (BDT)', 'Price', 'price', 'Fare', 'fare']
    for col in price_candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No price column found. Available columns: {df.columns.tolist()}")

def get_date_column(df: pd.DataFrame) -> str:
    """Detect the date column in the dataset."""
    date_candidates = ['Departure Date & Time', 'Date_of_Journey', 'Date', 'date']
    for col in date_candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No date column found. Available columns: {df.columns.tolist()}")

def summarize_fares(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and return summary statistics of fares by airline, source, destination."""
    price_col = get_price_column(df)
    
    summary = df.groupby('Airline')[price_col].agg(['mean', 'median', 'std', 'min', 'max', 'count']).round(2)
    summary.columns = ['Mean', 'Median', 'Std', 'Min', 'Max', 'Count']
    summary = summary.sort_values('Mean', ascending=False)
    
    logger.info(f"Fare summary by airline:\n{summary}")
    
    return summary

def plot_fare_distribution(df: pd.DataFrame, column: str = None) -> None:
    """Plot histogram of fare distribution with mean/median lines."""
    price_col = get_price_column(df) if column is None else column
    
    if price_col not in df.columns:
        logger.warning(f"Column '{price_col}' not found")
        return
    
    plt.figure(figsize=(12, 6))
    sns.histplot(df[price_col], kde=True, bins=50, color='skyblue')
    plt.axvline(df[price_col].mean(), color='red', linestyle='--', label=f'Mean: {df[price_col].mean():.2f}')
    plt.axvline(df[price_col].median(), color='green', linestyle='--', label=f'Median: {df[price_col].median():.2f}')
    plt.title(f'Distribution of {price_col}')
    plt.xlabel(price_col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Fare distribution plotted. Mean: {df[price_col].mean():.2f}, Median: {df[price_col].median():.2f}")

def plot_fare_by_airline(df: pd.DataFrame) -> None:
    """Bar chart of average fare by airline. Sort descending."""
    if 'Airline' not in df.columns:
        logger.warning("Airline column not found")
        return
    
    price_col = get_price_column(df)
    
    airline_avg = df.groupby('Airline')[price_col].mean().sort_values(ascending=True)
    
    plt.figure(figsize=(12, 8))
    airline_avg.plot(kind='barh', color='steelblue')
    plt.title('Average Fare by Airline')
    plt.xlabel('Average Price')
    plt.ylabel('Airline')
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Highest avg fare: {airline_avg.idxmax()} at {airline_avg.max():.2f}")

def plot_fare_boxplots(df: pd.DataFrame) -> None:
    """Boxplots showing fare variation across airlines."""
    if 'Airline' not in df.columns:
        logger.warning("Airline column not found")
        return
    
    price_col = get_price_column(df)
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=df, x='Airline', y=price_col, palette='Set2')
    plt.xticks(rotation=45, ha='right')
    plt.title('Fare Variation Across Airlines')
    plt.xlabel('Airline')
    plt.ylabel(price_col)
    plt.tight_layout()
    plt.show()

def plot_fare_by_season(df: pd.DataFrame) -> None:
    """Boxplot or bar chart of fare variation by month/season."""
    date_col = get_date_column(df)
    
    if 'Seasonality' in df.columns:
        month_avg = df.groupby('Seasonality')[get_price_column(df)].mean()
        
        plt.figure(figsize=(10, 6))
        month_avg.plot(kind='bar', color='coral')
        plt.title('Average Fare by Seasonality')
        plt.xlabel('Seasonality')
        plt.ylabel('Average Price')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Seasonal fare variation plotted.")
    else:
        df['Month'] = pd.to_datetime(df[date_col], errors='coerce').dt.month
        if df['Month'].isnull().all():
            logger.warning("Could not parse dates for seasonal analysis")
            return
        
        month_avg = df.groupby('Month')[get_price_column(df)].mean()
        
        plt.figure(figsize=(10, 6))
        month_avg.plot(kind='bar', color='coral')
        plt.title('Average Fare by Month')
        plt.xlabel('Month')
        plt.ylabel('Average Price')
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
        plt.tight_layout()
        plt.show()
        
        logger.info(f"Seasonal fare variation plotted. Highest month: {month_avg.idxmax()} at {month_avg.max():.2f}")

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Correlation heatmap for numerical features."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        logger.warning("No numerical columns found for correlation")
        return
    
    plt.figure(figsize=(10, 8))
    corr_matrix = numeric_df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.8:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr:
        logger.info(f"Highly correlated pairs (>0.8): {high_corr}")

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute and log: avg fare per airline, popular routes, seasonal variation."""
    kpis = {}
    price_col = get_price_column(df)
    
    if 'Airline' in df.columns:
        kpis['avg_fare_per_airline'] = df.groupby('Airline')[price_col].mean().to_dict()
        kpis['most_expensive_airline'] = df.groupby('Airline')[price_col].mean().idxmax()
        kpis['cheapest_airline'] = df.groupby('Airline')[price_col].mean().idxmin()
    
    if 'Source' in df.columns and 'Destination' in df.columns:
        df['Route_Combined'] = df['Source'] + ' -> ' + df['Destination']
        route_counts = df['Route_Combined'].value_counts()
        kpis['most_popular_route'] = str(route_counts.idxmax())
        kpis['popular_route_count'] = int(route_counts.max())
        
        route_prices = df.groupby('Route_Combined')[price_col].mean().sort_values(ascending=False)
        kpis['top_5_expensive_routes'] = {str(k): v for k, v in route_prices.head(5).to_dict().items()}
    
    if 'Seasonality' in df.columns:
        kpis['seasonal_variation'] = df.groupby('Seasonality')[price_col].mean().to_dict()
    
    if 'Days Before Departure' in df.columns:
        kpis['avg_fare_by_days_before_departure'] = df.groupby('Days Before Departure')[price_col].mean().to_dict()
    
    for key, value in kpis.items():
        safe_value = str(value).replace('→', '->').replace('–', '-')
        logger.info(f"KPI - {key}: {safe_value}")
    
    return kpis

def run_full_eda(df: pd.DataFrame) -> dict:
    """Run all EDA functions. Return KPI dict."""
    logger.info("Starting full EDA...")
    
    summarize_fares(df)
    plot_fare_distribution(df)
    plot_fare_by_airline(df)
    plot_fare_boxplots(df)
    plot_fare_by_season(df)
    plot_correlation_heatmap(df)
    kpis = compute_kpis(df)
    
    logger.info("Full EDA completed")
    return kpis
