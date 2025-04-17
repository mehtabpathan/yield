import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.api import VAR

# --- Parameters ---
data_filename = "synthetic_ois_data.csv"
# All columns are endogenous in the VAR model
all_cols = [
    "EFFR",
    "OIS_1m1m",
    "OIS_2m1m",
    "OIS_3m3m",
    "OIS_1y1y",
    "OIS_2y1y",
    "OIS_3y1y",
]
# Target columns for specific evaluation
target_cols_eval = ["OIS_3m3m", "OIS_1y1y", "OIS_2y1y", "OIS_3y1y"]
test_size = 0.2  # Fraction of data to use for testing
max_lags = 10  # Maximum lag order to check for VAR model selection
lag_selection_criterion = (
    "aic"  # Criterion for lag selection ('aic', 'bic', 'hqic', 'fpe')
)
forecast_confidence_level = 0.95  # For prediction intervals
alpha = 1 - forecast_confidence_level  # Alpha for forecast_interval

# --- Load Data ---
print(f"Loading data from {data_filename}...")
df = pd.read_csv(data_filename, index_col="Date", parse_dates=True)

# --- Preprocessing ---
# Calculate daily differences (changes)
print("Calculating daily rate changes...")
df_diff = df[all_cols].diff().dropna()  # Use all columns

# --- Train/Test Split (Chronological) ---
print(f"Splitting data (test_size={test_size})...")
n_obs = len(df_diff)
split_index = int(n_obs * (1 - test_size))
train_data = df_diff[:split_index]
test_data = df_diff[split_index:]

print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

# --- Lag Order Selection ---
print(
    f"Selecting optimal lag order (max_lags={max_lags}, criterion={lag_selection_criterion})..."
)
# Initialize VAR model on training data to select lag order
model_for_lag_selection = VAR(train_data)
lag_order_results = model_for_lag_selection.select_order(maxlags=max_lags)
selected_lag_order = lag_order_results.selected_orders[lag_selection_criterion]
# Ensure lag order is at least 1 to avoid issues with VAR(0) forecasting
selected_lag_order = max(1, selected_lag_order)
print(f"Selected lag order (min 1 enforced): {selected_lag_order}")
# print(lag_order_results.summary()) # Uncomment for detailed lag selection summary

# --- Model Training ---
print(f"Training VAR model with lag order p={selected_lag_order}...")
model = VAR(train_data)
model_fitted = model.fit(selected_lag_order)

# print(model_fitted.summary()) # Uncomment for detailed model summary

# --- Forecasting (One-Step-Ahead) ---
print("Generating one-step-ahead forecasts and prediction intervals on the test set...")
forecast_horizon = 1  # Predict only the next step
num_test_samples = len(test_data)
num_vars = len(all_cols)
forecasts = np.zeros((num_test_samples, num_vars))
forecast_lower = np.zeros((num_test_samples, num_vars))
forecast_upper = np.zeros((num_test_samples, num_vars))

# Iterate through the test set
for i in range(num_test_samples):
    # Get the lagged data required for the forecast
    # Use data from the combined train+test up to the current point
    current_data_index = split_index + i
    input_data = df_diff.iloc[
        current_data_index - selected_lag_order : current_data_index
    ]

    # Forecast the next step with interval
    point_forecast, lower_bound, upper_bound = model_fitted.forecast_interval(
        y=input_data.values, steps=forecast_horizon, alpha=alpha
    )
    forecasts[i, :] = point_forecast[0]  # Get the first (and only) step forecast
    forecast_lower[i, :] = lower_bound[0]
    forecast_upper[i, :] = upper_bound[0]

# Convert forecasts and bounds to DataFrames for easier handling
forecast_df = pd.DataFrame(forecasts, index=test_data.index, columns=all_cols)
forecast_lower_df = pd.DataFrame(
    forecast_lower, index=test_data.index, columns=all_cols
)
forecast_upper_df = pd.DataFrame(
    forecast_upper, index=test_data.index, columns=all_cols
)

# --- Evaluation ---
print("\n--- Model Evaluation (Predicting Daily Changes) ---")
results = {}
target_indices = [all_cols.index(col) for col in target_cols_eval]

for i, target_name in enumerate(target_cols_eval):
    target_idx = target_indices[i]
    actual = test_data.iloc[:, target_idx]
    predicted = forecast_df.iloc[:, target_idx]

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    results[target_name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"Target: {target_name}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  R2:   {r2:.4f}")

# --- Plotting Results ---
print(
    f"\nGenerating plots with {int(forecast_confidence_level*100)}% prediction intervals..."
)
num_targets_plot = len(target_cols_eval)
fig, axes = plt.subplots(
    num_targets_plot, 1, figsize=(12, num_targets_plot * 4), sharex=True
)
if num_targets_plot == 1:
    axes = [axes]

test_dates = test_data.index

for i, target_name in enumerate(target_cols_eval):
    target_idx = target_indices[i]
    ax = axes[i]
    ax.plot(
        test_dates,
        test_data.iloc[:, target_idx],
        label="Actual Change",
        linewidth=1.5,
        alpha=0.8,
    )
    ax.plot(
        test_dates,
        forecast_df.iloc[:, target_idx],
        label="Forecasted Change",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
    )
    # Add prediction interval shading
    ax.fill_between(
        test_dates,
        forecast_lower_df.iloc[:, target_idx],
        forecast_upper_df.iloc[:, target_idx],
        color="gray",
        alpha=0.3,
        label=f"{int(forecast_confidence_level*100)}% Prediction Interval",
    )
    ax.set_ylabel("Daily Change (%)")
    ax.set_title(
        f"Actual vs. Forecasted Daily Change (VAR): {target_name} (R2={results[target_name]['R2']:.3f})"
    )
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

axes[-1].set_xlabel("Date")
fig.suptitle("VAR Model: Test Set Performance (Daily Changes)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

print("\nScript finished.")
