import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings(
    "ignore", category=FutureWarning
)  # Ignore numpy future warnings from bootstrapping

# --- Parameters ---
data_filename = "synthetic_ois_data.csv"
base_feature_cols = ["EFFR", "OIS_1m1m", "OIS_2m1m"]
target_cols = ["OIS_3m3m", "OIS_1y1y", "OIS_2y1y", "OIS_3y1y"]
test_size = 0.2
lags_to_include = [1, 2, 3]
rolling_window_size = 5
n_bootstraps = 1000  # Number of bootstrap samples to generate per step
quantiles_to_calculate = [0.05, 0.50, 0.95]  # Quantiles from bootstrap distribution
median_quantile = 0.50

# --- Load Data ---
print(f"Loading data from {data_filename}...")
df = pd.read_csv(data_filename, index_col="Date", parse_dates=True)

# --- Preprocessing ---
print("Calculating daily rate changes...")
df_diff = df.diff().dropna()

# --- Feature Engineering (Keep the comprehensive feature set) ---
print("Engineering features...")
features_df = pd.DataFrame(index=df_diff.index)
all_feature_names = []
# ... (Paste the full feature engineering code again) ...
# 1. Add lagged features
print("  Adding lagged features...")
for lag in lags_to_include:
    for col in base_feature_cols:
        feature_name = f"{col}_lag{lag}"
        features_df[feature_name] = df_diff[col].shift(lag)
        all_feature_names.append(feature_name)
# 2. Add rolling mean features
print("  Adding rolling mean features...")
for col in base_feature_cols:
    feature_name = f"{col}_roll{rolling_window_size}_mean"
    features_df[feature_name] = (
        df_diff[col].rolling(window=rolling_window_size).mean().shift(1)
    )
    all_feature_names.append(feature_name)
# 3. Add rolling std dev features
print("  Adding rolling std dev features...")
for col in base_feature_cols:
    feature_name = f"{col}_roll{rolling_window_size}_std"
    features_df[feature_name] = (
        df_diff[col].rolling(window=rolling_window_size).std().shift(1)
    )
    all_feature_names.append(feature_name)
# 4. Add spread features
print("  Adding spread features...")
spread_2m1m_1m1m_diff = df_diff["OIS_2m1m"] - df_diff["OIS_1m1m"]
spread_1m1m_effr_diff = df_diff["OIS_1m1m"] - df_diff["EFFR"]
features_df["spread_2m1m_1m1m_diff_lag1"] = spread_2m1m_1m1m_diff.shift(1)
features_df["spread_1m1m_effr_diff_lag1"] = spread_1m1m_effr_diff.shift(1)
all_feature_names.extend(["spread_2m1m_1m1m_diff_lag1", "spread_1m1m_effr_diff_lag1"])
# 5. Add lagged target features
print("  Adding lagged target features...")
for col in target_cols:
    feature_name = f"{col}_lag1"
    features_df[feature_name] = df_diff[col].shift(1)
    all_feature_names.append(feature_name)
# 6. Add slope change feature
print("  Adding slope change feature...")
slope = (df["OIS_2m1m"] - df["EFFR"]) / 2
slope_change = slope.diff()
features_df["slope_2m1m_effr_change_lag1"] = slope_change.shift(1)
all_feature_names.append("slope_2m1m_effr_change_lag1")
# --- End Feature Engineering ---

y = df_diff[target_cols]
print("Aligning features and targets by dropping NaNs...")
original_len = len(features_df)
features_df = features_df.dropna()
nan_dropped_count = original_len - len(features_df)
print(f"Dropped {nan_dropped_count} initial rows.")
X = features_df
y_aligned = y.loc[X.index]
assert len(X) == len(y_aligned), "Alignment error"
print(f"Final samples: {len(X)}")

# --- Walk-Forward Validation with Residual Bootstrapping ---
print(
    f"Starting Walk-Forward Validation with Residual Bootstrapping (test_size={test_size}, n_bootstraps={n_bootstraps})..."
)
n_samples = len(X)
split_index = int(n_samples * (1 - test_size))
n_test_samples = n_samples - split_index
print(
    f"Total samples: {n_samples}, Training start size: {split_index}, Test steps: {n_test_samples}"
)

# Store bootstrapped quantiles and actuals
walk_forward_quantiles = {q: [] for q in quantiles_to_calculate}
actual_values = []

for i in range(split_index, n_samples):
    X_train_step = X.iloc[:i]
    y_train_step = y_aligned.iloc[:i]
    X_predict_step = X.iloc[[i]]
    y_actual_step = y_aligned.iloc[i]

    # 1. Train the base model
    model_step = LinearRegression()
    model_step.fit(X_train_step, y_train_step)

    # 2. Get point prediction
    y_pred_point = model_step.predict(X_predict_step)[0]

    # 3. Calculate residuals on training data for this step
    y_train_pred = model_step.predict(X_train_step)
    residuals = y_train_step.values - y_train_pred

    # Handle case with insufficient residuals for bootstrapping early on
    if len(residuals) < 1:
        # Cannot bootstrap, just use point prediction for all quantiles
        bootstrap_step_quantiles = {q: y_pred_point for q in quantiles_to_calculate}
    else:
        # 4. Perform bootstrapping
        bootstrap_predictions = np.zeros((n_bootstraps, len(target_cols)))
        for b in range(n_bootstraps):
            # Sample a residual vector WITH REPLACEMENT
            random_residual_idx = np.random.randint(0, len(residuals))
            sampled_residual = residuals[random_residual_idx]
            bootstrap_predictions[b, :] = y_pred_point + sampled_residual

        # 5. Calculate quantiles from bootstrap distribution
        bootstrap_step_quantiles = {}
        for q in quantiles_to_calculate:
            bootstrap_step_quantiles[q] = np.percentile(
                bootstrap_predictions, q * 100, axis=0
            )

    # Store quantiles for this step
    for q in quantiles_to_calculate:
        walk_forward_quantiles[q].append(bootstrap_step_quantiles[q])

    actual_values.append(y_actual_step.values)

    if (i - split_index + 1) % 50 == 0:  # Progress update
        print(
            f"  Completed step {i+1}/{n_samples} (Test step {i - split_index + 1}/{n_test_samples})"
        )

# Convert lists to numpy arrays
quantile_arrays = {
    q: np.array(walk_forward_quantiles[q]) for q in quantiles_to_calculate
}
y_test_walk_forward = np.array(actual_values)
test_dates = X.index[split_index:]
print("Walk-Forward Validation finished.")

# --- Evaluation (on Median Forecast) ---
print(
    f"\n--- Model Evaluation (Walk-Forward {int(median_quantile*100)}% Bootstrap Quantile Predictions) ---"
)
results = {}
y_pred_median = quantile_arrays[median_quantile]
for i, target_name in enumerate(target_cols):
    actual = y_test_walk_forward[:, i]
    predicted = y_pred_median[:, i]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)  # R2 on median is meaningful here
    results[target_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"Target: {target_name}")
    print(f"  RMSE (Median): {rmse:.6f}")
    print(f"  MAE (Median):  {mae:.6f}")
    print(f"  R2 (Median):   {r2:.4f}")

# --- Plotting Results (Bootstrap Quantiles) ---
print("\nGenerating plots for bootstrap quantile forecasts...")
num_targets = len(target_cols)
fig, axes = plt.subplots(num_targets, 1, figsize=(12, num_targets * 4), sharex=True)
if num_targets == 1:
    axes = [axes]
lower_quantile = min(quantiles_to_calculate)
upper_quantile = max(quantiles_to_calculate)
pred_lower = quantile_arrays[lower_quantile]
pred_upper = quantile_arrays[upper_quantile]
interval_label = f"{int(upper_quantile*100 - lower_quantile*100)}% Interval ({lower_quantile:.2f}-{upper_quantile:.2f})"

for i, target_name in enumerate(target_cols):
    ax = axes[i]
    ax.plot(
        test_dates,
        y_test_walk_forward[:, i],
        label="Actual Change",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        test_dates,
        y_pred_median[:, i],
        label=f"Median Forecast (Q{median_quantile:.2f})",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.fill_between(
        test_dates,
        pred_lower[:, i],
        pred_upper[:, i],
        color="gray",
        alpha=0.3,
        label=interval_label,
    )
    ax.set_ylabel("Daily Change (%)")
    ax.set_title(
        f"Bootstrap Walk-Forward: Actual vs. Forecasted Daily Change: {target_name} (R2={results[target_name]['R2']:.3f})"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

axes[-1].set_xlabel("Date")
fig.suptitle(
    "Residual Bootstrap + Linear Regression: Walk-Forward Performance", fontsize=16
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
print("\nScript finished.")
