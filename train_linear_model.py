import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --- Parameters ---
data_filename = "synthetic_ois_data.csv"
base_feature_cols = ["EFFR", "OIS_1m1m", "OIS_2m1m"]
target_cols = ["OIS_3m3m", "OIS_1y1y", "OIS_2y1y", "OIS_3y1y"]
test_size = 0.2  # Fraction of data to use for testing
lags_to_include = [1, 2, 3]  # Lags for base features
rolling_window_size = 5  # Window size for rolling mean/std

# --- Load Data ---
print(f"Loading data from {data_filename}...")
df = pd.read_csv(data_filename, index_col="Date", parse_dates=True)

# --- Preprocessing ---
# Calculate daily differences (changes)
print("Calculating daily rate changes...")
df_diff = df.diff().dropna()  # Drop the first row with NaNs

# --- Feature Engineering ---
print(
    "Engineering features (lags, rolling stats, spreads, lagged targets, slope change)..."
)
features_df = pd.DataFrame(index=df_diff.index)
all_feature_names = []

# 1. Add lagged features (from base_feature_cols)
print("  Adding lagged features...")
for lag in lags_to_include:
    for col in base_feature_cols:
        feature_name = f"{col}_lag{lag}"
        features_df[feature_name] = df_diff[col].shift(lag)
        all_feature_names.append(feature_name)

# 2. Add rolling mean features (from base_feature_cols)
print("  Adding rolling mean features...")
for col in base_feature_cols:
    feature_name = f"{col}_roll{rolling_window_size}_mean"
    features_df[feature_name] = (
        df_diff[col].rolling(window=rolling_window_size).mean().shift(1)
    )
    all_feature_names.append(feature_name)

# 3. Add rolling std dev features (from base_feature_cols)
print("  Adding rolling std dev features...")
for col in base_feature_cols:
    feature_name = f"{col}_roll{rolling_window_size}_std"
    features_df[feature_name] = (
        df_diff[col].rolling(window=rolling_window_size).std().shift(1)
    )
    all_feature_names.append(feature_name)

# 4. Add spread features (using differenced data, lagged)
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
    features_df[feature_name] = df_diff[col].shift(1)  # Lag 1 of target change
    all_feature_names.append(feature_name)

# 6. Add slope change feature
print("  Adding slope change feature...")
# Define slope using levels
slope = (df["OIS_2m1m"] - df["EFFR"]) / 2  # Arbitrary time diff=2
slope_change = slope.diff()
features_df["slope_2m1m_effr_change_lag1"] = slope_change.shift(1)
all_feature_names.append("slope_2m1m_effr_change_lag1")

# Define targets (y) using the original differenced data
y = df_diff[target_cols]

# Drop rows with NaNs created by lagging/rolling features
print("Aligning features and targets by dropping NaNs...")
original_len = len(features_df)
features_df = features_df.dropna()
nan_dropped_count = original_len - len(features_df)
print(f"Dropped {nan_dropped_count} initial rows due to feature generation NaNs.")

# Align y with the features_df (target time t corresponds to features calculated up to t-1)
X = features_df
y_aligned = y.loc[X.index]

# Ensure X and y have the same number of samples
assert len(X) == len(
    y_aligned
), "X and y must have the same number of samples after alignment"
print(f"Final number of samples for training/testing: {len(X)}")

# --- Walk-Forward Validation ---
print(f"Starting Walk-Forward Validation (test_size={test_size})...")
n_samples = len(X)
split_index = int(n_samples * (1 - test_size))
n_test_samples = n_samples - split_index

print(
    f"Total samples: {n_samples}, Training start size: {split_index}, Test steps: {n_test_samples}"
)

# Store predictions and actuals from the walk-forward process
walk_forward_predictions = []
actual_values = []

for i in range(split_index, n_samples):
    # Define current training data (up to step i-1)
    X_train_step = X.iloc[:i]
    y_train_step = y_aligned.iloc[:i]

    # Define the features for the step we want to predict (step i)
    X_predict_step = X.iloc[[i]]  # Use double brackets to keep DataFrame format
    y_actual_step = y_aligned.iloc[i]  # Actual value for this step

    # Train the model on data up to this point
    model_step = LinearRegression()
    model_step.fit(X_train_step, y_train_step)

    # Predict the single next step
    y_pred_step = model_step.predict(X_predict_step)

    # Store prediction and actual value
    walk_forward_predictions.append(y_pred_step[0])  # Get the single prediction array
    actual_values.append(y_actual_step.values)  # Get the actual values array

    if (i - split_index + 1) % 50 == 0:  # Print progress periodically
        print(
            f"  Completed step {i+1}/{n_samples} (Test step {i - split_index + 1}/{n_test_samples})"
        )

# Convert lists of predictions and actuals into numpy arrays for evaluation
y_pred_walk_forward = np.array(walk_forward_predictions)
y_test_walk_forward = np.array(actual_values)

# Get the dates corresponding to the test period for plotting
test_dates = X.index[split_index:]

print("Walk-Forward Validation finished.")

# --- Evaluation (on Walk-Forward CHANGE Predictions) ---
print("\n--- Model Evaluation (Walk-Forward 1-Step Ahead CHANGES) ---")
results_change = {}
for i, target_name in enumerate(target_cols):
    actual = y_test_walk_forward[:, i]
    predicted = y_pred_walk_forward[:, i]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    results_change[target_name] = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }
    print(f"Target: {target_name} (Change)")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.4f}")

# --- Convert Change Predictions to Level Predictions ---
print("\nConverting change predictions to level predictions...")
# Get the last actual level from the training period
last_train_feature_date = X.index[split_index - 1]
last_actual_levels = df.loc[last_train_feature_date, target_cols].values

# Calculate cumulative sum of predicted changes
predicted_change_cumsum = np.cumsum(y_pred_walk_forward, axis=0)

# Calculate predicted levels by adding cumulative changes to the last actual level
predicted_levels_array = last_actual_levels + predicted_change_cumsum
predicted_levels_df = pd.DataFrame(
    predicted_levels_array, index=test_dates, columns=target_cols
)

# Get actual levels for the test period
actual_levels_df = df.loc[test_dates, target_cols]

# --- Evaluation (on Walk-Forward LEVEL Predictions) ---
print("\n--- Model Evaluation (Walk-Forward 1-Step Ahead LEVELS) ---")
results_level = {}
for i, target_name in enumerate(target_cols):
    actual = actual_levels_df.iloc[:, i]
    predicted = predicted_levels_df.iloc[:, i]
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    # R2 is often misleadingly high for levels, focus on RMSE/MAE
    results_level[target_name] = {"RMSE": rmse, "MAE": mae}
    print(f"Target: {target_name} (Level)")
    print(f"  RMSE: {rmse:.6f}")  # Units are now rate levels (%)
    print(f"  MAE:  {mae:.6f}")  # Units are now rate levels (%)

# --- Plotting Results (Changes) ---
print("\nGenerating plots for walk-forward CHANGE predictions...")
num_targets = len(target_cols)
fig_change, axes_change = plt.subplots(
    num_targets, 1, figsize=(12, num_targets * 4), sharex=True
)
if num_targets == 1:
    axes_change = [axes_change]

for i, target_name in enumerate(target_cols):
    ax = axes_change[i]
    ax.plot(
        test_dates,
        y_test_walk_forward[:, i],
        label="Actual Change",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        test_dates,
        y_pred_walk_forward[:, i],
        label="Predicted Change (1-step ahead)",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.set_ylabel("Daily Change (%)")
    ax.set_title(
        f"Walk-Forward (Change): {target_name} (R2={results_change[target_name]['R2']:.3f})"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

axes_change[-1].set_xlabel("Date")
fig_change.suptitle(
    "Linear Regression (MANY Features): Walk-Forward Performance (Daily Changes)",
    fontsize=16,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
# plt.show() # Show separately or save

# --- Plotting Results (Levels) ---
print("\nGenerating plots for walk-forward LEVEL predictions...")
fig_level, axes_level = plt.subplots(
    num_targets, 1, figsize=(12, num_targets * 4), sharex=True
)
if num_targets == 1:
    axes_level = [axes_level]

for i, target_name in enumerate(target_cols):
    ax = axes_level[i]
    ax.plot(
        test_dates,
        actual_levels_df.iloc[:, i],
        label="Actual Level",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        test_dates,
        predicted_levels_df.iloc[:, i],
        label="Predicted Level (from 1-step changes)",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.set_ylabel("Rate Level (%)")
    ax.set_title(
        f"Walk-Forward (Level): {target_name} (RMSE={results_level[target_name]['RMSE']:.3f})"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

axes_level[-1].set_xlabel("Date")
fig_level.suptitle(
    "Linear Regression (MANY Features): Walk-Forward Performance (Levels)",
    fontsize=16,
)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()  # Show level plots (and implicitly change plots if not saved)

print("\nScript finished.")
