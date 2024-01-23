import numpy as np


X = np.random.randn(10, 2)
y = 0.1 + 0.1 * X[:,0] - 0.2 * X[:, 1] 

from sklearn.linear_model import Lasso

model = Lasso()
model.fit(X,y)

import joblib
joblib.dump(model, "regression_lineaire.joblib")