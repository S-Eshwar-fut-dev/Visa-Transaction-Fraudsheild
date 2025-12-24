import pandas as pd
import numpy as np
from typing import Tuple
import os

def load_and_split(path: str = 'data/raw/creditcard.csv') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ULB dataset, add IDs, time-split 80/10/10, return train/val/test."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Download creditcard.csv to {path}")
    
    df = pd.read_csv(path)
    df['Time'] = pd.to_datetime(df['Time'], unit='s') 
    
    df['user_id'] = (df[['V1', 'V2', 'Amount']].sum(axis=1).astype(int) % 50000).astype(int)
    df['merchant_id'] = (df['Class'] * df['Time'].astype(int) % 10000).astype(int)
    
    cutoffs = df['Time'].quantile([0.8, 0.9])
    train = df[df['Time'] < cutoffs[0.8]].copy()
    val = df[(df['Time'] >= cutoffs[0.8]) & (df['Time'] < cutoffs[0.9])].copy()
    test = df[df['Time'] >= cutoffs[0.9]].copy()
    
    np.random.seed(42)
    ring_users = np.random.choice(train['user_id'].unique(), 3)
    ring_merchants = np.random.choice(train['merchant_id'].unique(), 2)
    ring_data = {
        'Time': pd.date_range(start=train['Time'].min(), periods=500, freq='30S'),
        'V1': np.random.normal(2, 0.5, 500),
        'V2': np.random.normal(-2, 0.5, 500),
        'V4': np.random.uniform(0, 10, 500),  # Lat
        'V11': np.random.uniform(0, 10, 500),  # Lon
        'Amount': np.random.uniform(500, 2000, 500),
        'Class': 1,
        'user_id': np.random.choice(ring_users, 500),
        'merchant_id': np.random.choice(ring_merchants, 500)
    }
    synthetic = pd.DataFrame(ring_data)
    train = pd.concat([train, synthetic], ignore_index=True)

    train.to_csv('data/processed/train_augmented.csv', index=False)
    val.to_csv('data/processed/val.csv', index=False)
    test.to_csv('data/processed/test.csv', index=False)
    
    return train, val, test

if __name__ == "__main__":
    train, val, test = load_and_split()
    print(f"Train shape: {train.shape}, Fraud rate: {train['Class'].mean():.4f}")