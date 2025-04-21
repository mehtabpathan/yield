import pandas as pd
from sklearn.preprocessing import StandardScaler

# Use previously created synthetic dataset: df with columns x1 to x8
# We will only engineer features for x1, x2, x3 (input variables)

# Compute raw levels
levels = df[['x1', 'x2', 'x3']]

# Compute differences (1-day difference)
diffs = levels.diff().fillna(0).add_suffix('_diff')

# Moving averages (7-day, 30-day)
ma_7 = levels.rolling(window=7, min_periods=1).mean().add_suffix('_ma7')
ma_30 = levels.rolling(window=30, min_periods=1).mean().add_suffix('_ma30')

# Percentage changes
pct_changes = levels.pct_change().replace([np.inf, -np.inf], 0).fillna(0).add_suffix('_pct')

# Interaction terms (level * diff)
interactions = levels.values * diffs.values
interactions = pd.DataFrame(interactions, columns=[f'{col}_x_diff' for col in levels.columns])

# Z-score normalization of levels
zscaler = StandardScaler()
zscore_levels = pd.DataFrame(zscaler.fit_transform(levels), columns=[f'{col}_z' for col in levels.columns])

# Log-levels and log-differences
log_levels = np.log1p(levels).add_suffix('_log')
log_diffs = log_levels.diff().fillna(0).add_suffix('_diff')

# Percentile ranks (level quantiles)
quantile_levels = levels.rank(pct=True).add_suffix('_quantile')

# Gap-to-peak/trough (current level - 30-day max/min)
rolling_max = levels.rolling(30, min_periods=1).max()
rolling_min = levels.rolling(30, min_periods=1).min()
gap_to_peak = levels - rolling_max
gap_to_trough = levels - rolling_min
gap_to_peak.columns = [f'{col}_gap_peak' for col in levels.columns]
gap_to_trough.columns = [f'{col}_gap_trough' for col in levels.columns]

# Threshold indicators (e.g., above or below mean)
thresholds = (levels > levels.mean()).astype(int).add_suffix('_above_mean')

# Combine all engineered features
engineered_features = pd.concat([
    levels,
    diffs,
    ma_7,
    ma_30,
    pct_changes,
    interactions,
    zscore_levels,
    log_levels,
    log_diffs,
    quantile_levels,
    gap_to_peak,
    gap_to_trough,
    thresholds
], axis=1)

tools.display_dataframe_to_user(name="Engineered Features for x1, x2, x3", dataframe=engineered_features.head(10))
