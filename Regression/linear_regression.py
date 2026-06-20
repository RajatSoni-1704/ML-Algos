import numpy as np
import random

class simple_linear_regressor:
    def __init__(self):
        self.m = None
        self.b = None
    
    
    def fit(self, X:np.array, y:np.array):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        num = sum([(y[i] - y_mean)*(X[i] - X_mean) for i in range(len(X))])
        denom = sum([(X[i]-X_mean)**2 for i in range(len(X))])
        self.m = num/(denom)
        self.b = y_mean - (X_mean*self.m)
        return
    
    def predict(self, X_new:np.array):
        y_new = X_new*self.m + self.b
        return y_new
    
class multi_linear_regression:
    def __init__(self):
        self.__theta = None
        
    def fit(self, X:np.array, y:np.array):
        X = np.insert(X, 0, 1, axis=1)
        term1 = np.linalg.inv(X.T@X)
        self.__theta = term1@X.T@y
        return
    
    def predict(self, X:np.array):
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    
class BGDRegressor:
    def __init__(self, learning_rate:int, epochs:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X:np.array, y:np.array):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        X = np.insert(X, 0, 1, axis=1)
        n, m = X.shape
        y = y.reshape(-1, 1)
        self.__theta = np.random.rand(m, 1)
        for _ in range(self.epochs):
            diff = X@self.__theta - y
            cost = sum((diff)**2)/n
            self.__cost_values.append(cost)
            gradient = (2/n)*(X.T@diff)
            gradient = gradient.reshape(-1, 1)
            self.__theta = self.__theta - self.learning_rate*gradient
        return

    def predict(self, X:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    
    def error_values(self):
        return self.__cost_values
    
class SGDRegressor:
    def __init__(self, learning_rate:int, epochs:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        
    def learning_rate(self, t):
        t0, t1 = 5, 50
        return (t0/(t+t1))
    
    def fit(self, X:np.array, y:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                gradient = 2*(X_rand.T@diff)
                gradient = gradient.reshape(-1, 1)
                lr = self.learning_rate(i*m+j)
                self.__theta = self.__theta - gradient*lr
        return
    
    def predict(self, X:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    
class MBGDRegressor:
    def __init__(self, learning_rate:int, epoch:int, batch_size:int):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
    
    def fit(self, X:np.array, y:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epoch):
            for j in range(n//self.batch_size):
                idx = random.sample(range(n), self.batch_size)
                X_rand = X[idx]
                y_rand = y[idx].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                gradient = 2*(X_rand.T@diff)
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    
    
class RidgeRegression:
    def __init__(self, learning_rate:int, epochs:int, alpha:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        
    def fit(self, X:np.array, y:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                gradient = 2*(X_rand.T@diff) + 2*self.alpha*self.__theta
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta
    

class LassoRegression:
    def __init__(self, learning_rate:int, epochs:int, alpha:int):
        self.__theta = None
        self.__cost_values = []
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        
    def fit(self, X:np.array, y:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        n, m = np.shape(X)
        y = y.reshape(-1, 1)
        self.__theta = np.random.rand(m, 1)
        for i in range(self.epochs):
            for j in range(m):
                random_index = np.random.randint(low=0, high=n)
                X_rand = X[random_index].reshape(1, -1)
                y_rand = y[random_index].reshape(-1, 1)
                diff = X_rand@self.__theta - y_rand
                gradient = 2*(X_rand.T@diff) + 2*self.alpha
                gradient = gradient.reshape(-1, 1)
                # lr = self.learning_rate(i*m+j)
                self.__theta = self.__theta - gradient*self.learning_rate
        return
    
    def predict(self, X:np.array):
        X = (X - np.mean(X, axis=0))/np.std(X)
        X = np.insert(X, 0, 1, axis=1)
        return X@self.__theta