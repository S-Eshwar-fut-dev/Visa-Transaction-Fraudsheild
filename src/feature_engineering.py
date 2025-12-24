import pandas as pd
import numpy as np
from typing import Dict, Tuple
from math import radians, sin, cos, sqrt, atan2

def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in km."""
    R = 6371
    dlon, dlat = radians(lon2 - lon1), radians(lat2 - lat1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def engineer_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Engineer velocity, geo, fingerprint (train stats only)."""
    def safe_rolling(group, col, window, shift=True):
        rolled = group[col].rolling(window, closed='left', min_periods=1).agg(['count', 'sum', 'nunique'])
        if shift: rolled = rolled.shift(1).fillna(0)
        return rolled

    # Train-only user stats
    train_sorted = train.sort_values(['user_id', 'Time'])
    user_stats = train_sorted.groupby('user_id').agg({
        'Amount': ['mean', 'std'],
        'merchant_id': lambda x: x.nunique()
    }).reset_index()
    user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_merchant_diversity']
    user_stats = user_stats.fillna(0)

    dfs = {'train': train_sorted, 'val': val.sort_values(['user_id', 'Time']), 'test': test.sort_values(['user_id', 'Time'])}
    
    for key in dfs:
        df = dfs[key].copy()
        
        # Velocity (per user, past only)
        df['count_1h'] = safe_rolling(df, 'Time', '1H')['count']
        df['sum_1h'] = safe_rolling(df, 'Amount', '1H')['sum']
        df['unique_merchant_1h'] = safe_rolling(df, 'merchant_id', '1H')['nunique']
        
        # Geo (group-apply for speed)
        def geo_calc(g):
            dists = np.zeros(len(g))
            imposs = np.zeros(len(g))
            times = g['Time'].values
            lats, lons = g['V4'].values, g['V11'].values
            for i in range(1, len(g)):
                if pd.isna(times[i-1]): continue
                time_h = (times[i] - times[i-1]).total_seconds() / 3600
                dist = haversine(lons[i-1], lats[i-1], lons[i], lats[i])
                dists[i] = dist
                speed = dist / time_h if time_h > 0 else 0
                imposs[i] = 1 if speed > 1000 else 0
            return pd.Series({'dist_prev_km': dists, 'is_impossible_travel': imposs})
        geo_feats = df.groupby('user_id').apply(geo_calc).reset_index(level=0, drop=True)
        df = pd.concat([df, geo_feats], axis=1)
        
        # Fingerprint (merge train stats)
        df = df.merge(user_stats, on='user_id', how='left').fillna(0)
        df['amount_zscore'] = (df['Amount'] - df['user_avg_amount']) / (df['user_std_amount'] + 1e-8)
        df['merchant_novelty'] = (df.groupby('user_id')['merchant_id'].shift(1) != df['merchant_id']).astype(int)
        
        dfs[key] = df
    
    # Save for notebooks
    for key, df in dfs.items():
        df.to_csv(f'data/processed/{key}.csv', index=False)
    
    return dfs['train'], dfs['val'], dfs['test']

if __name__ == "__main__":
    from load_data import load_and_split
    train, val, test = load_and_split()
    train, val, test = engineer_features(train, val, test)
    print("Features engineered: Sample zscore", train['amount_zscore'].describe())