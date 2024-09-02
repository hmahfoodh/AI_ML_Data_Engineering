import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression

# Generate a synthetic dataset with noise
X, y = make_regression(n_samples=100, n_features=20, noise=5, random_state=0)

# Create L1 and L2 regularization models
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=0.1)

# Fit the models to the data
lasso.fit(X, y)
ridge.fit(X, y)

# Get the coefficients
lasso_coef = lasso.coef_
ridge_coef = ridge.coef_

# Print the coefficients
print("Lasso coefficients:", lasso_coef)
print("Ridge coefficients:", ridge_coef)