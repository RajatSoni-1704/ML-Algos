from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from Regression.linear_regression import LassoRegression
from sklearn.metrics import root_mean_squared_error, r2_score


X, y = make_regression(n_samples=100, n_features=4, n_targets=1, noise=0.5, random_state=12)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

gd_reg = LassoRegression(learning_rate=0.01, epochs=100, alpha=0.00001)

gd_reg.fit(X_train, y_train)
gd_reg_y_pred = gd_reg.predict(X_test)

print("RMSE of gd_reg",root_mean_squared_error(y_test, gd_reg_y_pred))
print("R2 score of gd_reg", r2_score(y_test, gd_reg_y_pred))

from sklearn.linear_model import Lasso
lr = Lasso()
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)
print("RMSE of lr", root_mean_squared_error(y_test, lr_y_pred))
print("R2 score of lr", r2_score(y_test, lr_y_pred))