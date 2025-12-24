import pandas as pd

def safe_rolling(group: pd.Series, window: str, agg: str = 'count', shift: bool = True) -> pd.Series:
    """Leakage-safe rolling with shift."""
    rolled = group.rolling(window, closed='left', min_periods=1).agg(agg)
    if shift: rolled = rolled.shift(1).fillna(0)
    return rolled