import numpy as np
import pandas as pd


def gaussian_kernel(x, target, sigma=1.0):
    """Maps distance to similarity via Gaussian kernel."""
    return np.exp(-0.5 * ((x - target) / sigma) ** 2)


def compute_slope(x):
    """Compute linear slope via least squares."""
    x_vals = np.arange(len(x))
    return np.polyfit(x_vals, x, 1)[0]


def apply_constraints(
    df, constraints, window_size=5, sigma=1.0, combine_method="geometric"
):
    """
    df: DataFrame with columns as variable names.
    constraints: dict of constraints per variable, e.g.:
        {
            'var1': {'min': 2, 'max': 8, 'slope_min': 0.1},
            'var3': {'slope_max': -0.2}
        }
    window_size: for computing slope
    sigma: controls fuzziness
    combine_method: one of ['geometric', 'arithmetic', 'min']

    Returns:
        Series of final sample weights.
    """
    n_rows = len(df)
    scores = np.ones(n_rows)

    for var, rules in constraints.items():
        var_scores = np.ones(n_rows)
        values = df[var].values

        # Apply min/max constraints on levels
        if "min" in rules:
            d = np.maximum(0, rules["min"] - values)
            var_scores *= gaussian_kernel(d, 0, sigma)

        if "max" in rules:
            d = np.maximum(0, values - rules["max"])
            var_scores *= gaussian_kernel(d, 0, sigma)

        # Apply slope constraints
        if "slope_min" in rules or "slope_max" in rules:
            slopes = np.full(n_rows, np.nan)
            for i in range(n_rows - window_size + 1):
                window = values[i : i + window_size]
                slopes[i + window_size - 1] = compute_slope(window)

            if "slope_min" in rules:
                d = np.maximum(0, rules["slope_min"] - slopes)
                slope_score = gaussian_kernel(d, 0, sigma)
                var_scores *= np.nan_to_num(slope_score, nan=1.0)

            if "slope_max" in rules:
                d = np.maximum(0, slopes - rules["slope_max"])
                slope_score = gaussian_kernel(d, 0, sigma)
                var_scores *= np.nan_to_num(slope_score, nan=1.0)

        # Combine with overall score
        if combine_method == "geometric":
            scores *= var_scores
        elif combine_method == "arithmetic":
            scores += var_scores
        elif combine_method == "min":
            scores = np.minimum(scores, var_scores)

    if combine_method == "arithmetic":
        scores /= len(constraints)

    return pd.Series(scores, index=df.index)


# Sample data
np.random.seed(0)
df = pd.DataFrame(
    {
        "var1": np.random.normal(5, 1, 100),
        "var2": np.random.normal(3, 1, 100),
        "var3": np.cumsum(np.random.normal(0, 0.1, 100)),  # trending variable
        "var4": np.random.uniform(0, 10, 100),
        "var5": np.sin(np.linspace(0, 10, 100)),
    }
)

# Define constraints
constraints = {
    "var1": {"min": 4, "max": 6},
    "var3": {"slope_min": 0.05},
    "var5": {"slope_max": 0.0},
}

# Apply constraint scoring
weights = apply_constraints(
    df, constraints, window_size=5, sigma=0.5, combine_method="geometric"
)
df["weight"] = weights
