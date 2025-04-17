import numpy as np
import pandas as pd

# --- Parameters ---
num_years = 15
days_per_year = 252  # Approximate trading days
num_days = num_years * days_per_year
start_date = pd.to_datetime("2009-01-01")
output_filename = "synthetic_ois_data.csv"

# Initial rate values (in percent) - chosen somewhat arbitrarily
initial_rates = {
    "EFFR": 1.5,
    "OIS_1m1m": 1.6,
    "OIS_2m1m": 1.7,
    "OIS_3m3m": 1.8,
    "OIS_1y1y": 2.2,
    "OIS_2y1y": 2.5,
    "OIS_3y1y": 2.8,
}
rate_names = list(initial_rates.keys())
num_rates = len(rate_names)

# --- Simulation Parameters ---
# Define standard deviations for daily shocks (basis points)
stdevs = np.array([1.5, 1.8, 2.0, 2.5, 3.5, 4.0, 4.5]) / 100  # Convert bps to percent

# Define a correlation matrix - high correlation at front, decreasing with tenor
corr_matrix = np.array(
    [
        [1.00, 0.98, 0.95, 0.90, 0.75, 0.70, 0.65],  # EFFR
        [0.98, 1.00, 0.97, 0.92, 0.80, 0.75, 0.70],  # 1m1m
        [0.95, 0.97, 1.00, 0.96, 0.85, 0.80, 0.75],  # 2m1m
        [0.90, 0.92, 0.96, 1.00, 0.90, 0.85, 0.80],  # 3m3m
        [0.75, 0.80, 0.85, 0.90, 1.00, 0.95, 0.90],  # 1y1y
        [0.70, 0.75, 0.80, 0.85, 0.95, 1.00, 0.96],  # 2y1y
        [0.65, 0.70, 0.75, 0.80, 0.90, 0.96, 1.00],  # 3y1y
    ]
)

# Convert correlation matrix and standard deviations to a covariance matrix
cov_matrix = np.outer(stdevs, stdevs) * corr_matrix

# Ensure covariance matrix is positive semi-definite (adjust if needed)
min_eigenvalue = np.min(np.linalg.eigvalsh(cov_matrix))
if min_eigenvalue < 0:
    cov_matrix -= (min_eigenvalue - 1e-6) * np.identity(
        num_rates
    )  # Add small value to diagonal

# --- Generate Data ---
# Generate correlated daily shocks
mean_shock = np.zeros(num_rates)
shocks = np.random.multivariate_normal(mean_shock, cov_matrix, size=num_days)

# Initialize rates array
rates = np.zeros((num_days, num_rates))
rates[0, :] = list(initial_rates.values())

# Simulate using random walk
for i in range(1, num_days):
    rates[i, :] = rates[i - 1, :] + shocks[i, :]
    # Apply a floor (e.g., 0%) to prevent negative rates
    rates[i, :] = np.maximum(0, rates[i, :])

# --- Create DataFrame ---
dates = pd.bdate_range(start=start_date, periods=num_days)  # Use business days
df = pd.DataFrame(rates, columns=rate_names, index=dates)
df.index.name = "Date"

# --- Save to CSV ---
df.to_csv(output_filename)

print(f"Generated synthetic data for {num_days} days.")
print(f"Saved data to {output_filename}")
print("First 5 rows:")
print(df.head())
print("Last 5 rows:")
print(df.tail())
