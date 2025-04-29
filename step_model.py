import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Load data
df = pd.read_csv('step_function_data_with_budget.csv')

# Prepare feature sets for comparison
# 1. Base features only
X_base = df[['X0', 'X1', 'X2', 'X3']]

# 2. With budget constraint feature
X_budget = df[['X0', 'X1', 'X2', 'X3', 'target_budget']]

# 3. With sign features
X_sign = df[['X0', 'X1', 'X2', 'X3']]
# Add sign features
X_sign = X_sign.assign(
    X1_pos=(df['X1'] > 0).astype(int),
    X2_pos=(df['X2'] > 0).astype(int),
    X3_pos=(df['X3'] > 0).astype(int),
    sign_count=lambda x: x.X1_pos + x.X2_pos + x.X3_pos
)

# 4. Combined features (both budget and sign)
X_combined = X_budget.copy()
X_combined = X_combined.assign(
    X1_pos=(df['X1'] > 0).astype(int),
    X2_pos=(df['X2'] > 0).astype(int),
    X3_pos=(df['X3'] > 0).astype(int),
    sign_count=lambda x: x.X1_pos + x.X2_pos + x.X3_pos,
    # Add remaining budget feature
    remaining_budget=df['target_budget'] - df[['X1', 'X2', 'X3']].sum(axis=1)
)

# Output variables
y = df[['X4', 'X5', 'X6', 'X7', 'X8']]

# Split data
X_base_train, X_base_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=42)
X_budget_train, X_budget_test, _, _ = train_test_split(X_budget, y, test_size=0.2, random_state=42)
X_sign_train, X_sign_test, _, _ = train_test_split(X_sign, y, test_size=0.2, random_state=42)
X_combined_train, X_combined_test, _, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Define models
def create_models():
    return {
        'Linear Regression': MultiOutputRegressor(LinearRegression()),
        'Ridge Regression': MultiOutputRegressor(Ridge(alpha=1.0)),
        'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        'Gradient Boosting': MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42))
    }

# Function for budget-constrained prediction
def budget_constrained_predictions(model, X, X_with_budget=None):
    """Apply budget constraint to model predictions"""
    if X_with_budget is None:
        X_with_budget = X
        
    # Get raw predictions
    y_pred = model.predict(X)
    
    # Get budget information
    if 'target_budget' in X_with_budget.columns:
        total_budget = X_with_budget['target_budget'].values
        used_budget = X_with_budget[['X1', 'X2', 'X3']].sum(axis=1).values
        remaining_budget = total_budget - used_budget
        
        # Scale predictions to match budget
        y_pred_sum = np.sum(y_pred, axis=1)
        scaling_factor = remaining_budget / y_pred_sum
        
        # Handle division by zero/infinity
        scaling_factor[np.isnan(scaling_factor) | np.isinf(scaling_factor)] = 1.0
        
        # Apply scaling
        y_pred_constrained = np.zeros_like(y_pred)
        for i in range(len(y_pred)):
            y_pred_constrained[i] = y_pred[i] * scaling_factor[i]
            
        return y_pred_constrained
    else:
        # Can't apply constraint without budget info
        return y_pred

# Training and evaluation function
def fit_and_evaluate(model_name, model, X_train, X_test, y_train, y_test, apply_constraint=False, X_budget_test=None):
    # Fit model
    model.fit(X_train, y_train)
    
    # Make predictions
    if apply_constraint and X_budget_test is not None:
        y_pred = budget_constrained_predictions(model, X_test, X_budget_test)
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Check sign prediction accuracy
    sign_accuracy = np.mean(np.sign(y_pred) == np.sign(y_test.values))
    
    # Calculate budget adherence if possible
    if X_budget_test is not None:
        total_budget = X_budget_test['target_budget'].values
        used_budget = X_budget_test[['X1', 'X2', 'X3']].sum(axis=1).values
        remaining_budget = total_budget - used_budget
        
        pred_sum = np.sum(y_pred, axis=1)
        budget_error = np.abs(remaining_budget - pred_sum).mean()
    else:
        budget_error = np.nan
    
    return {
        'Model': model_name,
        'MSE': mse,
        'R2': r2,
        'Sign Accuracy': sign_accuracy,
        'Budget Error': budget_error
    }

# Create results dataframe
results = []

# Base models
base_models = create_models()
for name, model in base_models.items():
    res = fit_and_evaluate(
        f"{name} (Base)", model, 
        X_base_train, X_base_test, 
        y_train, y_test
    )
    results.append(res)

# Budget-aware models
budget_models = create_models()
for name, model in budget_models.items():
    res = fit_and_evaluate(
        f"{name} (Budget)", model, 
        X_budget_train, X_budget_test, 
        y_train, y_test,
        apply_constraint=False, X_budget_test=X_budget_test
    )
    results.append(res)

# Budget-constrained models
budget_constrained_models = create_models()
for name, model in budget_constrained_models.items():
    res = fit_and_evaluate(
        f"{name} (Budget+Constrained)", model, 
        X_budget_train, X_budget_test, 
        y_train, y_test,
        apply_constraint=True, X_budget_test=X_budget_test
    )
    results.append(res)

# Sign-aware models
sign_models = create_models()
for name, model in sign_models.items():
    res = fit_and_evaluate(
        f"{name} (Sign)", model, 
        X_sign_train, X_sign_test, 
        y_train, y_test,
        apply_constraint=False, X_budget_test=X_budget_test
    )
    results.append(res)

# Combined models
combined_models = create_models()
for name, model in combined_models.items():
    res = fit_and_evaluate(
        f"{name} (Combined)", model, 
        X_combined_train, X_combined_test, 
        y_train, y_test,
        apply_constraint=False, X_budget_test=X_combined_test
    )
    results.append(res)

# Combined + constrained models
combined_constrained_models = create_models()
for name, model in combined_constrained_models.items():
    res = fit_and_evaluate(
        f"{name} (Combined+Constrained)", model, 
        X_combined_train, X_combined_test, 
        y_train, y_test,
        apply_constraint=True, X_budget_test=X_combined_test
    )
    results.append(res)

# Convert to dataframe for easy comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('MSE')

print("Model Comparison Results:")
print(results_df)

# Create visualization of model comparison
plt.figure(figsize=(15, 10))
sns.barplot(x='MSE', y='Model', data=results_df)
plt.title('Model Comparison - MSE (lower is better)')
plt.tight_layout()
plt.savefig('model_comparison_mse.png')
plt.close()

plt.figure(figsize=(15, 10))
sns.barplot(x='Sign Accuracy', y='Model', data=results_df)
plt.title('Model Comparison - Sign Prediction Accuracy (higher is better)')
plt.xlim([0, 1])
plt.tight_layout()
plt.savefig('model_comparison_sign_accuracy.png')
plt.close()

plt.figure(figsize=(15, 10))
# Filter out models without budget error
budget_results = results_df[~results_df['Budget Error'].isna()]
sns.barplot(x='Budget Error', y='Model', data=budget_results)
plt.title('Model Comparison - Budget Error (lower is better)')
plt.tight_layout()
plt.savefig('model_comparison_budget_error.png')
plt.close()

# Find best model
best_model_idx = results_df['MSE'].idxmin()
best_model_info = results_df.loc[best_model_idx]
best_model_name = best_model_info['Model']

print(f"\nBest model based on MSE: {best_model_name}")
print(f"MSE: {best_model_info['MSE']:.4f}")
print(f"R2: {best_model_info['R2']:.4f}")
print(f"Sign Accuracy: {best_model_info['Sign Accuracy']:.4f}")
print(f"Budget Error: {best_model_info['Budget Error']:.4f}")

# Let's analyze feature importance for the best combined model
if 'Combined' in best_model_name and 'Gradient Boosting' in best_model_name:
    best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    
    # Get feature importances for each target variable
    feature_importances = {}
    for i, col in enumerate(['X4', 'X5', 'X6', 'X7', 'X8']):
        best_model.fit(X_combined_train, y_train[col])
        feature_importances[col] = best_model.feature_importances_
    
    # Create a dataframe for visualization
    importance_df = pd.DataFrame(feature_importances, index=X_combined_train.columns)
    
    # Plot feature importances
    plt.figure(figsize=(15, 10))
    importance_df.plot(kind='bar', figsize=(15, 10))
    plt.title('Feature Importances for Each Target Variable')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig('feature_importances.png')
    plt.close()
    
    # Average importance across targets
    avg_importance = importance_df.mean(axis=1).sort_values(ascending=False)
    print("\nAverage Feature Importance:")
    print(avg_importance)
    
    plt.figure(figsize=(12, 8))
    avg_importance.plot(kind='bar', figsize=(12, 8))
    plt.title('Average Feature Importance Across All Targets')
    plt.ylabel('Importance')
    plt.xlabel('Features')
    plt.tight_layout()
    plt.savefig('avg_feature_importance.png')
    plt.close()

# Let's analyze specific patterns in sign persistence
# Create sign pattern groups
def get_sign_pattern(row):
    signs = []
    for col in ['X1', 'X2', 'X3']:
        if row[col] > 0:
            signs.append('+')
        elif row[col] < 0:
            signs.append('-')
        else:
            signs.append('0')
    return ''.join(signs)

# Add sign pattern to test data
X_combined_test_with_pattern = X_combined_test.copy()
X_combined_test_with_pattern['sign_pattern'] = X_combined_test.apply(
    lambda row: get_sign_pattern(row), axis=1
)

# Get predictions from best model
best_model_idx = results_df['MSE'].idxmin()
best_model_info = results_df.loc[best_model_idx]
best_model_name = best_model_info['Model']

if 'Combined+Constrained' in best_model_name:
    model_type = best_model_name.split(' ')[0]
    if model_type == 'Gradient':
        model_type = 'Gradient Boosting'
    
    model = eval(f"{model_type}Regressor(n_estimators=100, random_state=42)")
    model = MultiOutputRegressor(model)
    model.fit(X_combined_train, y_train)
    
    y_pred = budget_constrained_predictions(model, X_combined_test, X_combined_test)
    
    # Add predictions to dataframe
    for i, col in enumerate(['X4', 'X5', 'X6', 'X7', 'X8']):
        X_combined_test_with_pattern[f'pred_{col}'] = y_pred[:, i]
        X_combined_test_with_pattern[f'actual_{col}'] = y_test[col].values
        X_combined_test_with_pattern[f'error_{col}'] = np.abs(X_combined_test_with_pattern[f'pred_{col}'] - X_combined_test_with_pattern[f'actual_{col}'])
        
    # Group by sign pattern and analyze errors
    pattern_analysis = X_combined_test_with_pattern.groupby('sign_pattern').agg({
        'error_X4': 'mean',
        'error_X5': 'mean',
        'error_X6': 'mean',
        'error_X7': 'mean',
        'error_X8': 'mean'
    }).reset_index()
    
    # Add average error
    pattern_analysis['avg_error'] = pattern_analysis[['error_X4', 'error_X5', 'error_X6', 'error_X7', 'error_X8']].mean(axis=1)
    pattern_analysis = pattern_analysis.sort_values('avg_error')
    
    print("\nError by Sign Pattern:")
    print(pattern_analysis)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='sign_pattern', y='avg_error', data=pattern_analysis)
    plt.title('Average Prediction Error by Sign Pattern')
    plt.xlabel('Sign Pattern (X1, X2, X3)')
    plt.ylabel('Average Error')
    plt.tight_layout()
    plt.savefig('error_by_sign_pattern.png')
    plt.close()
    
    # Analyze sign accuracy by pattern
    for i, col in enumerate(['X4', 'X5', 'X6', 'X7', 'X8']):
        X_combined_test_with_pattern[f'sign_match_{col}'] = (
            np.sign(X_combined_test_with_pattern[f'pred_{col}']) == 
            np.sign(X_combined_test_with_pattern[f'actual_{col}'])
        ).astype(int)
    
    sign_acc_by_pattern = X_combined_test_with_pattern.groupby('sign_pattern').agg({
        'sign_match_X4': 'mean',
        'sign_match_X5': 'mean',
        'sign_match_X6': 'mean',
        'sign_match_X7': 'mean',
        'sign_match_X8': 'mean'
    }).reset_index()
    
    sign_acc_by_pattern['avg_sign_accuracy'] = sign_acc_by_pattern[['sign_match_X4', 'sign_match_X5', 'sign_match_X6', 'sign_match_X7', 'sign_match_X8']].mean(axis=1)
    sign_acc_by_pattern = sign_acc_by_pattern.sort_values('avg_sign_accuracy', ascending=False)
    
    print("\nSign Prediction Accuracy by Pattern:")
    print(sign_acc_by_pattern)
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='sign_pattern', y='avg_sign_accuracy', data=sign_acc_by_pattern)
    plt.title('Sign Prediction Accuracy by Sign Pattern')
    plt.xlabel('Sign Pattern (X1, X2, X3)')
    plt.ylabel('Sign Accuracy (0-1)')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('sign_accuracy_by_pattern.png')
    plt.close()

print("\nAnalysis complete!")
