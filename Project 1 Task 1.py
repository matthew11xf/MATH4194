# Task 1: Simple Linear Regression on Simulated Data

import numpy as np
import matplotlib.pyplot as pp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to generate synthetic data
def generate_linear_data(N, var):
    # Create a random number generator
    rng = np.random.default_rng()
    # Create an ordered list of x values
    X = rng.uniform(0, 10, N)
    X.sort()
    # Calculate y values with noise
    y = 3*X + rng.normal(0, var, N)
    return X.reshape(-1, 1), y

# Different noise levels to test
noise_levels = [0.1, 0.5, 2.0]

# Initialize plot
pp.figure(figsize=(12, 6))

for i, noise in enumerate(noise_levels, 1):
    # Generate synthetic data
    X, y = generate_linear_data(100, noise)
    
    # Create 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Fit linear model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Compute Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Noise Level = {noise}, MSE = {mse: .4f}")
    
    # Plot data and regression line
    pp.subplot(1, 3, i)
    pp.scatter(X, y)
    pp.plot(X, model.predict(X), color='red')
    pp.title(f"Noise Level = {noise}")
    
pp.tight_layout()
pp.show()