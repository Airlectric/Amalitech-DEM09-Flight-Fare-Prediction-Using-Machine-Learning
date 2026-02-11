"""Data loading and initial inspection module."""

import pandas as pd
import os
from src.logger import get_logger

logger = get_logger(__name__)

def get_default_data_path() -> str:
    """Get the default path to the raw data folder (works from anywhere in project)."""
    # Navigate from this file's location to the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, 'data', 'raw', 'Flight_Price_Dataset_of_Bangladesh.csv')
    return data_path

def load_dataset(filepath: str = None) -> pd.DataFrame:
    """Load CSV dataset and log basic info. If no filepath provided, uses default path."""
    if filepath is None:
        filepath = get_default_data_path()
    
    if not os.path.exists(filepath):
        logger.error(f"Dataset file not found: {filepath}")
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def inspect_dataset(df: pd.DataFrame) -> dict:
    """
    Run and log: df.shape, df.info(), df.describe(), df.head(),
    missing value counts, duplicate row count, data types.
    Return a dict summary with keys: 'shape', 'dtypes', 'missing', 'duplicates', 'describe'.
    """
    logger.info("Inspecting dataset...")
    
    summary = {}
    summary['shape'] = df.shape
    summary['dtypes'] = df.dtypes.to_dict()
    summary['missing'] = df.isnull().sum().to_dict()
    summary['duplicates'] = df.duplicated().sum()
    
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Data types:\n{df.dtypes}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")
    
    return summary
