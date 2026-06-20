import numpy as np
import random

# From-scratch regression algorithms implemented with NumPy.
# These classes are written for learning the math behind each model.

# Simple linear regression for one input feature.
# It uses the closed-form equations for slope and intercept.
class simple_linear_regressor:
    def __init__(self):
        # m is the slope and b is the intercept in y = mX + b.
        self.m = None
        self.b = None
    
    
    def fit(self, X:np.array, y:np.array):
        # The slope is covariance(X, y) divided by variance(X).
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        num = sum([(y[i] - y_mean)*(X[i] - X_mean) for i in range(len(X))])
        denom = sum([(X[i]-X_mean)**2 for i in range(len(X))])
        self.m = num/(denom)
        # Once the slope is known, choose b so the line passes through the means.
        self.b = y_mean - (X_mean*self.m)
        return
    
    def predict(self, X_new:np.array):
        # Prediction follows the learned line equation.
        y_new = X_new*self.m + self.b
        return y_new

# Multiple linear regression using the normal equation.
# theta = (X.T X)^-1 X.T y gives the least-squares solution directly.
class multi_linear_regression:
    def __init__(self):
        # theta stores the intercept and feature weights together.
        self.__theta = None
        
    def fit(self, X:np.array, y:np.array):
        # Insert a leading column of 1s so theta[0] acts as the intercept.
        X = np.insert(X, 0, 1, axis=1)
        term1 = np.linalg.inv(X.T@X)
        self.__theta = term1@X.T@y
        return
    
    def predict(self, X:np.array):
        # Add the same intercept column before applying X @ theta.
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta

# Linear regression trained with batch gradient descent.
# Each update uses the gradient computed from the full training dataset.
class BGDRegressor:
    def __init__(self, learning_rate:int, epochs:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X:np.array, y:np.array):
        # Standardize features so gradient descent is more stable.
        X = (X - X.mean(axis=0))/X.std(axis=0)
        # Add the intercept column to learn bias with the other weights.
        X = np.insert(X, 0, 1, axis=1)
        n, m = X.shape
        y = y.reshape(-1, 1)
        # Start with random weights, then improve them over epochs.
        self.__theta = np.random.rand(m, 1)
        for _ in range(self.epochs):
            # Difference between predictions and true targets.
            diff = X@self.__theta - y
            # Mean squared error cost for tracking training progress.
            cost = sum((diff)**2)/n
            self.__cost_values.append(cost)
            # Gradient of mean squared error with respect to theta.
            gradient = (2/n)*(X.T@diff)
            gradient = gradient.reshape(-1, 1)
            # Move theta in the opposite direction of the gradient.
            self.__theta = self.__theta - self.learning_rate*gradient
        return

    def predict(self, X:np.array):
        # Apply the same standardization pattern before prediction.
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        # Matrix multiplication gives predictions for all rows.
        return X@self.__theta
    
    def error_values(self):
        return self.__cost_values

# Linear regression trained with stochastic gradient descent.
# Each update uses one randomly selected training example.
class SGDRegressor:
    def __init__(self, learning_rate:int, epochs:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def learning_rate(self, t):
        # Decay the learning rate as training progresses.
        t0, t1 = 5, 50
        return (t0/(t+t1))
    
    def fit(self, X:np.array, y:np.array):
        # Standardize features to keep SGD updates numerically stable.
        X = (X - np.mean(X, axis=0))/np.std(X)
        # Add intercept column for the bias term.
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        # Random initialization gives the optimizer a starting point.
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                # Pick one random row and update theta using only that sample.
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                # Single-sample gradient for squared error.
                gradient = 2*(X_rand.T@diff)
                gradient = gradient.reshape(-1, 1)
                lr = self.learning_rate(i*m+j)
                # Smaller learning rates later in training reduce oscillation.
                self.__theta = self.__theta - gradient*lr
        return
    
    def predict(self, X:np.array):
        # Prepare inputs in the same shape used during training.
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta

# Linear regression trained with mini-batch gradient descent.
# Each update uses a random subset of rows instead of one row or all rows.
class MBGDRegressor:
    def __init__(self, learning_rate:int, epoch:int, batch_size:int):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
    
    def fit(self, X:np.array, y:np.array):
        # Standardize features before gradient-based optimization.
        X = (X - np.mean(X, axis=0))/np.std(X)
        # Insert the intercept column.
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        # Initialize one parameter per column, including the intercept.
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epoch):
            for j in range(n//self.batch_size):
                # Randomly sample a mini-batch for this update.
                idx = random.sample(range(n), self.batch_size)
                X_rand = X[idx]
                y_rand = y[idx].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                # Mini-batch gradient approximates the full dataset gradient.
                gradient = 2*(X_rand.T@diff)
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                # Update weights using the mini-batch gradient.
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        # Standardize and add intercept before calculating X @ theta.
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    

# Ridge regression adds L2 regularization to the gradient update.
# The L2 penalty discourages very large weights.
class RidgeRegression:
    def __init__(self, learning_rate:int, epochs:int, alpha:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        
    def fit(self, X:np.array, y:np.array):
        # Feature scaling helps the regularized gradient update behave evenly.
        X = (X - np.mean(X, axis=0))/np.std(X)
        # Add bias column so the model can learn an intercept.
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        # Initialize all model parameters randomly.
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                # Use one random training row per update.
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                # Add the L2 regularization gradient: 2 * alpha * theta.
                gradient = 2*(X_rand.T@diff) + 2*self.alpha*self.__theta
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                # Update weights after combining error and regularization gradients.
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        # Prepare input features and predict with X @ theta.
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    

# Lasso regression adds an L1-style regularization term.
# The L1 penalty encourages smaller weights and can push some toward zero.
class LassoRegression:
    def __init__(self, learning_rate:int, epochs:int, alpha:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        
    def fit(self, X:np.array, y:np.array):
        # Standardize features so the penalty affects features more fairly.
        X = (X - np.mean(X, axis=0))/np.std(X)
        # Insert intercept column for the bias parameter.
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        # Initialize model parameters before iterative optimization.
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                # Select one random example for a stochastic-style update.
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                # Add the L1-style penalty term controlled by alpha.
                gradient = 2*(X_rand.T@diff) + 2*self.alpha
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                # Move parameters against the combined gradient.
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        # Standardize, add intercept, then calculate predictions.
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
