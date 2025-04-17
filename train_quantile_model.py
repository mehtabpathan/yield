import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Parameters ---
data_filename = "synthetic_ois_data.csv"
base_feature_cols = ["EFFR", "OIS_1m1m", "OIS_2m1m"]
target_cols = ["OIS_3m3m", "OIS_1y1y", "OIS_2y1y", "OIS_3y1y"]
test_size = 0.2
lags_to_include = [1, 2, 3]
rolling_window_size = 5
quantiles_to_predict = [0.05, 0.50, 0.95]
median_quantile = 0.50

# Gradient Boosting Parameters (can be tuned)
gb_params = {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.7}

# --- Load Data ---
print(f"Loading data from {data_filename}...")
df = pd.read_csv(data_filename, index_col="Date", parse_dates=True)

# --- Preprocessing ---
print("Calculating daily rate changes...")
df_diff = df.diff().dropna()

# --- Feature Engineering (Keep the comprehensive feature set) ---
print(
    "Engineering features (lags, rolling stats, spreads, lagged targets, slope change)..."
)
features_df = pd.DataFrame(index=df_diff.index)
all_feature_names = []
# ... (Paste the full feature engineering code from the previous version here) ...
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
# --- End of Feature Engineering paste ---

y = df_diff[target_cols]
print("Aligning features and targets by dropping NaNs...")
original_len = len(features_df)
features_df = features_df.dropna()
nan_dropped_count = original_len - len(features_df)
print(f"Dropped {nan_dropped_count} initial rows due to feature generation NaNs.")
X = features_df
y_aligned = y.loc[X.index]
assert len(X) == len(
    y_aligned
), "X and y must have the same number of samples after alignment"
print(f"Final number of samples for training/testing: {len(X)}")

# --- Walk-Forward Validation ---
print(f"Starting Walk-Forward Validation with Quantile GBR (test_size={test_size})...")
n_samples = len(X)
split_index = int(n_samples * (1 - test_size))
n_test_samples = n_samples - split_index
print(
    f"Total samples: {n_samples}, Training start size: {split_index}, Test steps: {n_test_samples}"
)

# Store predictions for each quantile and actuals
walk_forward_predictions = {q: [] for q in quantiles_to_predict}
actual_values = []

for i in range(split_index, n_samples):
    X_train_step = X.iloc[:i]
    y_train_step = y_aligned.iloc[:i]
    X_predict_step = X.iloc[[i]]
    y_actual_step = y_aligned.iloc[i]

    step_predictions = {q: {} for q in quantiles_to_predict}

    # Train separate model for each target and each quantile
    for target_idx, target_name in enumerate(target_cols):
        y_target_train_step = y_train_step.iloc[:, target_idx]

        for q in quantiles_to_predict:
            model_step = GradientBoostingRegressor(
                loss="quantile", alpha=q, **gb_params
            )
            model_step.fit(X_train_step, y_target_train_step)
            y_pred_quantile = model_step.predict(X_predict_step)
            step_predictions[q][target_name] = y_pred_quantile[0]

    # Store predictions for this step
    for q in quantiles_to_predict:
        # Collect predictions for all targets for this quantile
        quantile_preds = [
            step_predictions[q][target_name] for target_name in target_cols
        ]
        walk_forward_predictions[q].append(quantile_preds)

    actual_values.append(y_actual_step.values)

    if (i - split_index + 1) % 25 == 0:
        print(
            f"  Completed step {i+1}/{n_samples} (Test step {i - split_index + 1}/{n_test_samples})"
        )

# Convert lists to numpy arrays
pred_arrays = {q: np.array(walk_forward_predictions[q]) for q in quantiles_to_predict}
y_test_walk_forward = np.array(actual_values)
test_dates = X.index[split_index:]
print("Walk-Forward Validation finished.")

# --- Evaluation (on Median Forecast) ---
print(
    f"\n--- Model Evaluation (Walk-Forward {int(median_quantile*100)}% Quantile Predictions) ---"
)
results = {}
y_pred_median = pred_arrays[median_quantile]
for i, target_name in enumerate(target_cols):
    actual = y_test_walk_forward[:, i]
    predicted = y_pred_median[:, i]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    results[target_name] = {"RMSE": rmse, "MAE": mae}
    print(f"Target: {target_name}")
    print(f"  RMSE (Median): {rmse:.6f}")
    print(f"  MAE (Median):  {mae:.6f}")

# --- Plotting Results (Quantile Forecasts) ---
print("\nGenerating plots for quantile forecasts...")
num_targets = len(target_cols)
fig, axes = plt.subplots(num_targets, 1, figsize=(12, num_targets * 4), sharex=True)
if num_targets == 1:
    axes = [axes]
lower_quantile = min(quantiles_to_predict)
upper_quantile = max(quantiles_to_predict)
pred_lower = pred_arrays[lower_quantile]
pred_upper = pred_arrays[upper_quantile]
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
        f"Quantile GBR Walk-Forward: Actual vs. Forecasted Daily Change: {target_name}"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

axes[-1].set_xlabel("Date")
fig.suptitle(
    "Quantile Gradient Boosting: Walk-Forward Performance (Daily Changes)", fontsize=16
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()
print("\nScript finished.")
