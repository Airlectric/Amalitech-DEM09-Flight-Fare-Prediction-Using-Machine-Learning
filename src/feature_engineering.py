"""Feature engineering module."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from src.logger import get_logger

logger = get_logger(__name__)

def create_date_features(df: pd.DataFrame, date_col: str = 'Date_of_Journey') -> pd.DataFrame:
    """Extract from date column: Month, Day, Weekday, Season."""
    logger.info(f"Creating date features from '{date_col}'")
    
    if date_col not in df.columns:
        logger.error(f"Date column '{date_col}' not found")
        return df
    
    df['Month'] = df[date_col].dt.month
    df['Day'] = df[date_col].dt.day
    df['Weekday'] = df[date_col].dt.dayofweek
    df['DayOfYear'] = df[date_col].dt.dayofyear
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
    
    df['Season'] = df['Month'].apply(get_season)
    
    logger.info(f"Created date features: Month, Day, Weekday, DayOfYear, Season")
    return df

def encode_categorical_features(df: pd.DataFrame, strategy: str = 'onehot') -> pd.DataFrame:
    """Encode categorical columns using one-hot or label encoding."""
    logger.info(f"Encoding categorical features using '{strategy}' strategy")
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if 'Route' in categorical_cols:
        categorical_cols.remove('Route')
    
    logger.info(f"Categorical columns to encode: {categorical_cols}")
    
    if strategy == 'onehot':
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        logger.info(f"One-hot encoded {len(categorical_cols)} columns")
    elif strategy == 'label':
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            logger.info(f"Label encoded '{col}' with {len(le.classes_)} classes")
        df = df.drop(columns=categorical_cols)
    
    return df

def scale_numerical_features(df: pd.DataFrame, columns: list = None, scaler_type: str = 'standard') -> tuple:
    """Scale specified numerical columns. Return (scaled_df, fitted_scaler)."""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Price' in columns:
            columns.remove('Price')
    
    logger.info(f"Scaling {len(columns)} numerical features using '{scaler_type}' scaler")
    
    if scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    logger.info(f"Scaled features: {columns}")
    return df_scaled, scaler

def split_data(df: pd.DataFrame, target_col: str = 'Price', test_size: float = 0.2, random_state: int = 42) -> tuple:
    """Split into X_train, X_test, y_train, y_test."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    logger.info(f"Train/Test split: {X_train.shape[0]} / {X_test.shape[0]} samples")
    logger.info(f"Train ratio: {X_train.shape[0] / len(X) * 100:.1f}%, Test ratio: {X_test.shape[0] / len(X) * 100:.1f}%")
    
    return X_train, X_test, y_train, y_test

def run_feature_pipeline(df: pd.DataFrame, target_col: str = 'Price') -> tuple:
    """Master function: date features -> encoding -> scaling -> splitting."""
    logger.info("Starting feature engineering pipeline...")
    
    df = create_date_features(df)
    df = encode_categorical_features(df, strategy='onehot')
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    df_scaled, scaler = scale_numerical_features(df, columns=numeric_cols, scaler_type='standard')
    
    X_train, X_test, y_train, y_test = split_data(df_scaled, target_col=target_col)
    
    feature_names = X_train.columns.tolist()
    logger.info(f"Total features after engineering: {len(feature_names)}")
    
    logger.info("Feature engineering pipeline completed")
    return X_train, X_test, y_train, y_test, scaler, feature_names
