import numpy as np

class PCA:
    def __init__(self, n_components:int):
        self.n_components = n_components
        self.eigenvalues = None
        self.eigenvectors = None
        self.X_new = None
    
    def fit(self, X:np.array):
        X_mean = np.mean(X, axis=0)
        self.X_new = (X - X_mean)
        X_cov = np.cov(self.X_new.T)
        self.eigenvalues, self.eigenvector = np.linalg.eig(X_cov)
        return
    
    def get_features(self, X:np.array):
        self.linear_transformation(X)
        indices = np.argsort(self.eigenvalues)[::-1]
        self.eigenvalues = self.eigenvalues[indices]
        top_eigenvectors = self.eigenvector[indices][:self.n_components]
        X_final = np.dot(self.X_new, top_eigenvectors.T)
        return X_final
    
    def get_explained_variance_ratio(self):
        return (sum(self.eigenvalues[:self.n_components])/sum(self.eigenvalues))