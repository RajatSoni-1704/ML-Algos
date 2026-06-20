# Regression

This folder contains from-scratch implementations of regression algorithms using Python and NumPy. The goal is to understand how regression models work internally without relying on high-level machine learning libraries.

## Implemented Models

The current implementation is available in `linear_regression.py`.

### `simple_linear_regressor`

Implements simple linear regression for one independent variable using the closed-form slope and intercept equations.

Main methods:

- `fit(X, y)`: Learns the slope and intercept from the training data.
- `predict(X_new)`: Predicts target values for new input data.

### `multi_linear_regression`

Implements multiple linear regression using the normal equation.

Main methods:

- `fit(X, y)`: Learns model parameters using matrix operations.
- `predict(X)`: Predicts target values for input feature arrays.

### `BGDRegressor`

Implements linear regression using batch gradient descent.

Main methods:

- `fit(X, y)`: Trains the model by updating parameters using the full dataset in each epoch.
- `predict(X)`: Predicts target values.
- `error_values()`: Returns the recorded cost values during training.

### `SGDRegressor`

Implements linear regression using stochastic gradient descent.

Main methods:

- `fit(X, y)`: Updates model parameters using randomly selected training examples.
- `predict(X)`: Predicts target values.

### `MBGDRegressor`

Implements linear regression using mini-batch gradient descent.

Main methods:

- `fit(X, y)`: Updates model parameters using randomly sampled mini-batches.
- `predict(X)`: Predicts target values.

### `RidgeRegression`

Implements ridge regression with L2 regularization using gradient-based optimization.

Main methods:

- `fit(X, y)`: Trains the model with an L2 penalty controlled by `alpha`.
- `predict(X)`: Predicts target values.

### `LassoRegression`

Implements lasso regression with L1-style regularization using gradient-based optimization.

Main methods:

- `fit(X, y)`: Trains the model with a regularization penalty controlled by `alpha`.
- `predict(X)`: Predicts target values.

## Requirements

- Python
- NumPy

Install NumPy with:

```bash
pip install numpy