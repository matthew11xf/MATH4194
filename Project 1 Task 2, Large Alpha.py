# Task 2: Polynomial Regression and Regularization
# This file uses a large alpha for the Ridge and Lasso models

import numpy as np
import matplotlib.pyplot as pp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# Function to generate synthetic quadratic data
def generate_quadratic_data(N, var):
    # Create a random number generator
    rng = np.random.default_rng()
    # Create an ordered list of x values
    X = rng.uniform(0, 10, N)
    X.sort()
    # Calculate y values with noise
    y = 2*X**2 + 3*X + rng.normal(0, var, N)
    return X.reshape(-1, 1), y

# For this task, we'll only use one noise level
noise = 20.0
# And, we'll only use one large alpha value
large_alpha = 500.0

# Initialize plot
pp.figure(figsize=(12, 8))

# Generate data
X, y = generate_quadratic_data(100, noise)
    
# Create 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear model
linear_model = LinearRegression()
# Fit the linear model
linear_model.fit(X_train, y_train)
# Predict on the test set
y_pred_linear = linear_model.predict(X_test)
# Calculate mean squared error
mse_linear = mean_squared_error(y_test, y_pred_linear)
    
# Create quadratic polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=False)

# Create an unregularized quadratic model
quadratic_model = make_pipeline(poly_features, LinearRegression())
# Fit the quadratic model
quadratic_model.fit(X_train, y_train)
# Predict on the test set
y_pred_quadratic = quadratic_model.predict(X_test)
mse_quadratic = mean_squared_error(y_test, y_pred_quadratic)


# RIDGE REGRESSION
# Objective: ||Xw - y||^2 + alpha*||w||^2

# Create a ridge model with a large alpha
ridge_model = make_pipeline(poly_features, Ridge(alpha=large_alpha))
# Fit the ridge model
ridge_model.fit(X_train, y_train)

# Calculate predicted y values
y_pred_ridge = ridge_model.predict(X_test)
# Calculate mean squared error
mse_ridge = mean_squared_error(y_test, y_pred_ridge)


# LASSO REGRESSION
# Objective: (1/(2*N))*||Xw - y||^2 + alpha*|w|

# Create a lasso model with a large alpha
lasso_model = make_pipeline(poly_features, Lasso(alpha=large_alpha))
# Fit the lasso model
lasso_model.fit(X_train, y_train)

# Calculate predicted y values
y_pred_lasso = lasso_model.predict(X_test)
# Calculate mean squared error
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Print results
print(f"Noise Level = {noise}")
print(f"  Linear Model MSE = {mse_linear:.4f}")
print(f"  Quadratic Model MSE = {mse_quadratic:.4f}")
print(f"  Ridge Regression MSE = {mse_ridge:.4f}")
print(f"  Lasso Regression MSE = {mse_lasso:.4f}\n")

# Plot data and regression lines
pp.scatter(X, y, label='Synthetic Data')
pp.plot(X, linear_model.predict(X), color='red', label='Linear Regression')
pp.plot(X, quadratic_model.predict(X), color='green', label='Quadratic Regression')
pp.plot(X, ridge_model.predict(X), color='orange', linestyle='dashed', label='Ridge Quadratic')
pp.plot(X, lasso_model.predict(X), color='purple', linestyle='dashed', label='Lasso Quadratic')
pp.title(f"Noise Level = {noise}, Alpha = {large_alpha}")
pp.xlabel('x')
pp.ylabel('y')
pp.legend()
pp.tight_layout()
pp.show()
