"""Data cleaning and preprocessing module."""

import pandas as pd
import numpy as np
from src.logger import get_logger

logger = get_logger(__name__)

def drop_irrelevant_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Drop unnamed/index/irrelevant columns. Log which columns were dropped."""
    if columns is None:
        columns = [col for col in df.columns if 'unnamed' in col.lower() or col.lower() == 'index']
    
    if columns:
        logger.info(f"Dropping irrelevant columns: {columns}")
        df = df.drop(columns=columns, errors='ignore')
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numerical columns: median imputation
    - Categorical columns: mode imputation
    Log the number of imputed values per column.
    """
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
    """
    - Remove or replace negative fares/base fares.
    - Normalize inconsistent city names (e.g., 'Dhaka' vs 'Dacca').
    - Log all corrections made.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if 'fare' in col.lower() or 'price' in col.lower():
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                logger.warning(f"Found {negative_count} negative values in '{col}'. Setting to 0.")
                df.loc[df[col] < 0, col] = 0
    
    city_mappings = {
        'Dacca': 'Dhaka',
        'Calcutta': 'Kolkata',
        'Bombay': 'Mumbai',
        'Madras': 'Chennai',
        'Bangalore': 'Bengaluru'
    }
    
    city_cols = ['Source', 'Destination']
    for col in city_cols:
        if col in df.columns:
            for old_name, new_name in city_mappings.items():
                changes = (df[col] == old_name).sum()
                if changes > 0:
                    df[col] = df[col].replace(old_name, new_name)
                    logger.info(f"Normalized {changes} occurrences of '{old_name}' to '{new_name}' in '{col}'")
    
    return df

def validate_and_convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Convert fare columns to float.
    - Convert date columns to datetime.
    - Log dtype conversions.
    """
    fare_cols = [col for col in df.columns if 'fare' in col.lower() or 'price' in col.lower() or 'tax' in col.lower()]
    
    for col in fare_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
            logger.info(f"Converted '{col}' to float")
    
    date_cols = [col for col in df.columns if 'date' in col.lower()]
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
    """
    Master cleaning function that calls all above functions in sequence.
    Returns the cleaned DataFrame.
    Log the shape before and after cleaning.
    """
    logger.info(f"Starting data cleaning. Original shape: {df.shape}")
    
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = fix_invalid_entries(df)
    df = validate_and_convert_dtypes(df)
    df = remove_duplicates(df)
    
    logger.info(f"Data cleaning complete. Final shape: {df.shape}")
    return df
