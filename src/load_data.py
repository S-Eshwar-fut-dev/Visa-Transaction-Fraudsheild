# src/load_data.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoadError(Exception):
    """Custom exception for data loading errors."""
    pass

def validate_dataframe(df: pd.DataFrame, required_cols: list) -> None:
    """Validate DataFrame has required columns and non-empty."""
    if df.empty:
        raise DataLoadError("DataFrame is empty")
    
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise DataLoadError(f"Missing required columns: {missing}")
    
    logger.info(f"DataFrame validated: {len(df)} rows, {len(df.columns)} columns")

def load_and_split(
    path: str = 'data/raw/creditcard.csv',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    augment_ring: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load ULB dataset with strict time-based split and optional ring augmentation.
    
    Args:
        path: Path to creditcard.csv
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        random_seed: Random seed for reproducibility
        augment_ring: Whether to add synthetic fraud ring to training data
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Loading data from {path}")
    
    if not os.path.exists(path):
        raise DataLoadError(
            f"Data file not found at {path}. "
            f"Download from https://www.kaggle.com/mlg-ulb/creditcardfraud"
        )
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise DataLoadError(f"Failed to read CSV: {str(e)}")
    
    # Validate required columns
    required_cols = ['Time', 'Amount', 'Class'] + [f'V{i}' for i in range(1, 29)]
    validate_dataframe(df, required_cols)
    
    # Convert Time to datetime (seconds since first transaction)
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    logger.info(f"Time range: {df['Time'].min()} to {df['Time'].max()}")
    
    # Create synthetic user_id and merchant_id
    # Use hash-based approach for reproducibility
    np.random.seed(random_seed)
    df['user_id'] = (
        (df['V1'] * 1000 + df['V2'] * 100 + df['Amount']).abs().astype(int) % 50000
    ).astype(str)
    df['merchant_id'] = (
        (df['V3'] * 1000 + df['V4'] * 100).abs().astype(int) % 10000
    ).astype(str)
    
    # Time-based split
    cutoff_train = df['Time'].quantile(train_ratio)
    cutoff_val = df['Time'].quantile(train_ratio + val_ratio)
    
    train = df[df['Time'] < cutoff_train].copy()
    val = df[(df['Time'] >= cutoff_train) & (df['Time'] < cutoff_val)].copy()
    test = df[df['Time'] >= cutoff_val].copy()
    
    logger.info(f"Initial split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    logger.info(f"Train fraud rate: {train['Class'].mean():.4f}")
    logger.info(f"Val fraud rate: {val['Class'].mean():.4f}")
    logger.info(f"Test fraud rate: {test['Class'].mean():.4f}")
    
    # Add synthetic fraud ring to TRAIN ONLY (no leakage)
    if augment_ring:
        train = _augment_with_fraud_ring(train, random_seed)
        logger.info(f"After augmentation - Train: {len(train)}, fraud rate: {train['Class'].mean():.4f}")
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train.to_csv('data/processed/train_raw.csv', index=False)
    val.to_csv('data/processed/val_raw.csv', index=False)
    test.to_csv('data/processed/test_raw.csv', index=False)
    
    return train, val, test

def _augment_with_fraud_ring(
    train: pd.DataFrame,
    random_seed: int,
    n_users: int = 3,
    n_merchants: int = 2,
    n_transactions: int = 500
) -> pd.DataFrame:
    """
    Add synthetic fraud ring to training data.
    
    Creates coordinated fraud pattern:
    - Small set of users transacting with small set of merchants
    - High frequency, high amounts
    - Anomalous V1/V2 patterns
    - Concentrated geography
    """
    np.random.seed(random_seed)
    
    # Select existing users/merchants to avoid introducing entirely new IDs
    ring_users = np.random.choice(train['user_id'].unique(), min(n_users, len(train['user_id'].unique())), replace=False)
    ring_merchants = np.random.choice(train['merchant_id'].unique(), min(n_merchants, len(train['merchant_id'].unique())), replace=False)
    
    # Generate synthetic transactions with fraud characteristics
    time_start = train['Time'].min()
    time_end = train['Time'].min() + pd.Timedelta(hours=12)  # Concentrated in 12h window
    
    synthetic_data = {
        'Time': pd.date_range(start=time_start, end=time_end, periods=n_transactions),
        'V1': np.random.normal(2.5, 0.3, n_transactions),  # Anomalous pattern
        'V2': np.random.normal(-2.5, 0.3, n_transactions),
        'V3': np.random.normal(0, 0.5, n_transactions),
        'V4': np.random.uniform(35.0, 36.0, n_transactions),  # Concentrated lat
        'V5': np.random.normal(0, 1, n_transactions),
        'V6': np.random.normal(0, 1, n_transactions),
        'V7': np.random.normal(0, 1, n_transactions),
        'V8': np.random.normal(0, 1, n_transactions),
        'V9': np.random.normal(0, 1, n_transactions),
        'V10': np.random.normal(0, 1, n_transactions),
        'V11': np.random.uniform(-95.0, -94.0, n_transactions),  # Concentrated lon
        'V12': np.random.normal(0, 1, n_transactions),
        'V13': np.random.normal(0, 1, n_transactions),
        'V14': np.random.normal(0, 1, n_transactions),
        'V15': np.random.normal(0, 1, n_transactions),
        'V16': np.random.normal(0, 1, n_transactions),
        'V17': np.random.normal(0, 1, n_transactions),
        'V18': np.random.normal(0, 1, n_transactions),
        'V19': np.random.normal(0, 1, n_transactions),
        'V20': np.random.normal(0, 1, n_transactions),
        'V21': np.random.normal(0, 1, n_transactions),
        'V22': np.random.normal(0, 1, n_transactions),
        'V23': np.random.normal(0, 1, n_transactions),
        'V24': np.random.normal(0, 1, n_transactions),
        'V25': np.random.normal(0, 1, n_transactions),
        'V26': np.random.normal(0, 1, n_transactions),
        'V27': np.random.normal(0, 1, n_transactions),
        'V28': np.random.normal(0, 1, n_transactions),
        'Amount': np.random.uniform(800, 2500, n_transactions),  # High amounts
        'Class': 1,  # All fraudulent
        'user_id': np.random.choice(ring_users, n_transactions),
        'merchant_id': np.random.choice(ring_merchants, n_transactions)
    }
    
    synthetic_df = pd.DataFrame(synthetic_data)
    augmented = pd.concat([train, synthetic_df], ignore_index=True)
    augmented = augmented.sort_values('Time').reset_index(drop=True)
    
    logger.info(f"Added {n_transactions} synthetic ring transactions")
    return augmented

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    train, val, test = load_and_split()
    print(f"\nFinal shapes:")
    print(f"Train: {train.shape}, Fraud: {train['Class'].sum()}")
    print(f"Val: {val.shape}, Fraud: {val['Class'].sum()}")
    print(f"Test: {test.shape}, Fraud: {test['Class'].sum()}")