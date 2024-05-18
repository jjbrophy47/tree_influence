import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from tree_influence.explainers import BoostIn

X_train, X_test, y_train, y_test = load_csv('X_train'), load_csv('X_test'), load_csv('y_train'), load_csv('y_test')
model = XGBClassifier(tree_method='hist')
X_train_val, y_train_vals = X_train.values, y_train.values.squeeze()
X_test_val, y_test = X_test.values, y_test.values.squeeze()
model.fit(X_train, y_train)

# fit influence estimator
explainer = BoostIn().fit(model, X_train, y_train)