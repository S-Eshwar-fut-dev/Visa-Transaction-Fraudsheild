# src/feature_engineering.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from math import radians, sin, cos, sqrt, atan2
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate Haversine distance in kilometers between two lat/lon points.
    
    Args:
        lon1, lat1: First point coordinates
        lon2, lat2: Second point coordinates
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    # Handle NaN values
    if any(pd.isna([lon1, lat1, lon2, lat2])):
        return 0.0
    
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c

def compute_velocity_features(
    df: pd.DataFrame,
    windows: List[str] = ['1H', '3H', '24H']
) -> pd.DataFrame:
    """
    Compute rolling velocity features per user (STRICTLY no leakage).
    
    Uses closed='left' and shift(1) to ensure current transaction is excluded.
    
    Args:
        df: DataFrame sorted by user_id and Time
        windows: List of rolling window sizes (e.g., '1H', '3H')
        
    Returns:
        DataFrame with velocity features added
    """
    logger.info("Computing velocity features...")
    
    df = df.sort_values(['user_id', 'Time']).reset_index(drop=True)
    
    for window in windows:
        logger.info(f"  Processing window: {window}")
        
        # Set Time as index for rolling (per user)
        df_indexed = df.set_index('Time')
        
        # Group by user and compute rolling aggregations
        # closed='left' excludes right boundary (current timestamp)
        # shift(1) moves values down one row (excludes current transaction)
        grouped = df_indexed.groupby('user_id', group_keys=False)
        
        # Transaction count
        df[f'count_{window}'] = grouped['Amount'].rolling(
            window, closed='left', min_periods=0
        ).count().shift(1).fillna(0).values
        
        # Amount sum
        df[f'sum_{window}'] = grouped['Amount'].rolling(
            window, closed='left', min_periods=0
        ).sum().shift(1).fillna(0).values
        
        # Unique merchants
        df[f'unique_merchant_{window}'] = grouped['merchant_id'].rolling(
            window, closed='left', min_periods=1
        ).apply(lambda x: x.nunique(), raw=False).shift(1).fillna(0).values
        
        # Average amount
        df[f'avg_amount_{window}'] = grouped['Amount'].rolling(
            window, closed='left', min_periods=1
        ).mean().shift(1).fillna(0).values
    
    logger.info(f"Velocity features computed: {len([c for c in df.columns if any(w in c for w in windows)])} new columns")
    return df

def compute_geo_features(df: pd.DataFrame, speed_threshold_kmh: float = 1000) -> pd.DataFrame:
    """
    Compute geography-based features (distance, impossible travel).
    
    Args:
        df: DataFrame with V4 (latitude) and V11 (longitude)
        speed_threshold_kmh: Speed threshold for impossible travel flag
        
    Returns:
        DataFrame with geo features
    """
    logger.info("Computing geo features...")
    
    df = df.sort_values(['user_id', 'Time']).reset_index(drop=True)
    
    def process_user_geo(group):
        """Process geography for single user."""
        n = len(group)
        distances = np.zeros(n)
        speeds = np.zeros(n)
        impossible_flags = np.zeros(n, dtype=int)
        
        times = group['Time'].values
        lats = group['V4'].values
        lons = group['V11'].values
        
        for i in range(1, n):
            # Calculate time difference in hours
            time_diff = (times[i] - times[i-1]) / np.timedelta64(1, 'h')
            
            if time_diff <= 0 or pd.isna(time_diff):
                continue
            
            # Calculate distance
            dist_km = haversine(lons[i-1], lats[i-1], lons[i], lats[i])
            distances[i] = dist_km
            
            # Calculate speed
            speed_kmh = dist_km / time_diff
            speeds[i] = speed_kmh
            
            # Flag impossible travel
            if speed_kmh > speed_threshold_kmh:
                impossible_flags[i] = 1
        
        return pd.DataFrame({
            'dist_prev_km': distances,
            'speed_kmh': speeds,
            'is_impossible_travel': impossible_flags
        }, index=group.index)
    
    # Apply to each user with progress bar
    tqdm.pandas(desc="Processing users")
    geo_features = df.groupby('user_id', group_keys=False).progress_apply(process_user_geo)
    
    # Merge back
    df = df.join(geo_features)
    
    logger.info(f"Geo features computed: {geo_features['is_impossible_travel'].sum()} impossible travel flags")
    return df

def compute_user_fingerprint_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute user behavioral fingerprint features (TRAIN-ONLY stats).
    
    Critical: User statistics computed ONLY from training data and
    safely merged to validation/test sets with proper handling of unseen users.
    
    Args:
        train: Training set
        val: Validation set
        test: Test set
        
    Returns:
        Tuple of (train, val, test) with fingerprint features
    """
    logger.info("Computing user fingerprint features (train-only stats)...")
    
    # Compute user statistics from TRAIN ONLY
    user_stats = train.groupby('user_id').agg({
        'Amount': ['mean', 'std', 'min', 'max', 'count'],
        'merchant_id': lambda x: x.nunique()
    }).reset_index()
    
    user_stats.columns = [
        'user_id', 
        'user_avg_amount', 'user_std_amount', 'user_min_amount', 'user_max_amount', 'user_tx_count',
        'user_merchant_diversity'
    ]
    
    # Fill NaN std with 0 (single transaction users)
    user_stats['user_std_amount'] = user_stats['user_std_amount'].fillna(0)
    
    logger.info(f"Computed stats for {len(user_stats)} unique users from training set")
    
    # Define default values for unseen users
    defaults = {
        'user_avg_amount': train['Amount'].median(),
        'user_std_amount': 0,
        'user_min_amount': 0,
        'user_max_amount': 0,
        'user_tx_count': 0,
        'user_merchant_diversity': 0
    }
    
    # Merge to each split
    for name, df in [('train', train), ('val', val), ('test', test)]:
        # Left merge preserves all transactions
        df = df.merge(user_stats, on='user_id', how='left')
        
        # Fill unseen users with defaults
        for col, default_val in defaults.items():
            df[col] = df[col].fillna(default_val)
        
        # Compute derived features
        # Z-score of amount (how unusual is this transaction for this user)
        df['amount_zscore'] = (df['Amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-8)
        df['amount_zscore'] = df['amount_zscore'].clip(-10, 10)  # Cap extreme values
        
        # Amount ratio to user's max
        df['amount_to_max_ratio'] = df['Amount'] / (df['user_max_amount'] + 1e-8)
        df['amount_to_max_ratio'] = df['amount_to_max_ratio'].clip(0, 100)
        
        # Merchant novelty (is this a new merchant for this user?)
        df = df.sort_values(['user_id', 'Time']).reset_index(drop=True)
        df['merchant_novelty'] = (
            df.groupby('user_id')['merchant_id'].shift(1) != df['merchant_id']
        ).astype(int)
        df['merchant_novelty'] = df['merchant_novelty'].fillna(1)  # First transaction = novel
        
        # Update the df in our data
        if name == 'train':
            train = df
        elif name == 'val':
            val = df
        else:
            test = df
        
        logger.info(f"{name.capitalize()} set: {len(df)} transactions, "
                   f"{(df['user_id'].isin(user_stats['user_id'])).sum()} known users")
    
    return train, val, test

def engineer_features(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    windows: List[str] = ['1H', '3H'],
    speed_threshold: float = 1000
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Master function to engineer all features with strict no-leakage guarantees.
    
    Args:
        train: Training set
        val: Validation set
        test: Test set
        windows: Rolling window sizes
        speed_threshold: Speed threshold for impossible travel (km/h)
        
    Returns:
        Tuple of (train, val, test) with all features
    """
    logger.info("="*60)
    logger.info("Starting feature engineering pipeline")
    logger.info("="*60)
    
    # Validate inputs
    for name, df in [('train', train), ('val', val), ('test', test)]:
        if df.empty:
            raise ValueError(f"{name} set is empty")
        if not df['Time'].is_monotonic_increasing:
            logger.warning(f"{name} set not sorted by Time, sorting now")
            df.sort_values('Time', inplace=True)
    
    # 1. Velocity features (per-user rolling, no leakage)
    train = compute_velocity_features(train, windows)
    val = compute_velocity_features(val, windows)
    test = compute_velocity_features(test, windows)
    
    # 2. Geo features
    train = compute_geo_features(train, speed_threshold)
    val = compute_geo_features(val, speed_threshold)
    test = compute_geo_features(test, speed_threshold)
    
    # 3. User fingerprint features (train-only stats)
    train, val, test = compute_user_fingerprint_features(train, val, test)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    train.to_csv('data/processed/train_features.csv', index=False)
    val.to_csv('data/processed/val_features.csv', index=False)
    test.to_csv('data/processed/test_features.csv', index=False)
    
    logger.info("="*60)
    logger.info("Feature engineering complete")
    logger.info(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
    logger.info("="*60)
    
    return train, val, test

if __name__ == "__main__":
    import os
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    from load_data import load_and_split
    
    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    
    print("\nSample features:")
    feature_cols = [c for c in train.columns if any(x in c for x in ['count_', 'sum_', 'dist_', 'zscore', 'novelty'])]
    print(train[feature_cols].head())
    print("\nFeature statistics:")
    print(train[feature_cols].describe())