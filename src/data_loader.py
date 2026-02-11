"""Data loading and initial inspection module."""

import pandas as pd
import os
from src.logger import get_logger

logger = get_logger(__name__)

def load_dataset(filepath: str) -> pd.DataFrame:
    """Load CSV dataset and log basic info. Return the DataFrame."""
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
