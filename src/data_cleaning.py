"""Data cleaning and preprocessing module."""

import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def get_price_column(df: pd.DataFrame) -> str:
    """Detect the price/fare column in the dataset."""
    price_candidates = ['Total Fare (BDT)', 'Base Fare (BDT)', 'Price', 'price']
    for col in price_candidates:
        if col in df.columns:
            return col
    raise ValueError(f"No price column found. Available columns: {df.columns.tolist()}")

def drop_irrelevant_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Drop unnamed/index/irrelevant columns. Log which columns were dropped."""
    if columns is None:
        columns = [col for col in df.columns if 'unnamed' in col.lower() or col.lower() == 'index']
    
    if columns:
        logger.info(f"Dropping irrelevant columns: {columns}")
        df = df.drop(columns=columns, errors='ignore')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: numerical columns with median, categorical with mode."""
    missing_before = df.isnull().sum()
    logger.info(f"Missing values before imputation:\n{missing_before[missing_before > 0]}")
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                median_val = df[col].median()
                imputed_count = df[col].isnull().sum()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {imputed_count} missing values in '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                imputed_count = df[col].isnull().sum()
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed {imputed_count} missing values in '{col}' with mode")
    
    return df

def fix_invalid_entries(df: pd.DataFrame) -> pd.DataFrame:
    """Remove or replace negative fares. Normalize inconsistent city names."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    price_col = get_price_column(df)
    if price_col in numeric_cols:
        negative_count = (df[price_col] < 0).sum()
        if negative_count > 0:
            logger.warning(f"Found {negative_count} negative values in '{price_col}'. Setting to 0.")
            df.loc[df[price_col] < 0, price_col] = 0
    
    city_mappings = {
        'Dacca': 'Dhaka',
        'Calcutta': 'Kolkata',
        'Bombay': 'Mumbai',
        'Madras': 'Chennai',
        'Bangalore': 'Bengaluru',
        'Banglore': 'Bengaluru'
    }
    
    city_cols = ['Source', 'Source Name', 'Destination', 'Destination Name']
    for col in city_cols:
        if col in df.columns:
            for old_name, new_name in city_mappings.items():
                changes = (df[col] == old_name).sum()
                if changes > 0:
                    df[col] = df[col].replace(old_name, new_name)
                    logger.info(f"Normalized {changes} occurrences of '{old_name}' to '{new_name}' in '{col}'")
    
    return df

def validate_and_convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert fare columns to float, date columns to datetime."""
    price_col = get_price_column(df)
    if price_col in df.columns:
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce').astype(float)
        logger.info(f"Converted '{price_col}' to float")
    
    date_cols = ['Departure Date & Time', 'Date_of_Journey', 'Arrival Date & Time']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
            logger.info(f"Converted '{col}' to datetime")
    
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows. Log how many were removed."""
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df = df.drop_duplicates()
        logger.info(f"Removed {duplicates_before} duplicate rows")
    else:
        logger.info("No duplicate rows found")
    return df

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Master cleaning function. Returns the cleaned DataFrame."""
    logger.info(f"Starting data cleaning. Original shape: {df.shape}")
    
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = fix_invalid_entries(df)
    df = validate_and_convert_dtypes(df)
    df = remove_duplicates(df)
    
    logger.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df
