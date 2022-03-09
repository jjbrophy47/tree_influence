TreeInfluence: Influence Estimation for Gradient-Boosted Decision Trees
---
**tree_influence** is a python library that implements influence estimation for gradient-boosted decision trees (GBDTs), adapting popular techniques such as TracIn and Influence Functions to GBDTs. This library is compatible with all major GBDT frameworks including LightGBM, XGBoost, CatBoost, and SKLearn.

[Illustration]

Installation
---
1. Install Python 3.9.6+.
2. Run `make all`.

<!--```shell
pip install tree_influence
```-->

Usage
---
Simple example using *BoostIn* to identify the training examples that *most* decrease the loss of a given test instance:

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from tree_influence.explainers import BoostIn

# load iris data
data = load_iris()
X, y = data['data'], data['target']

# use only two classes, then split into train and test
idxs = np.where(y != 2)[0]
X, y = X[idxs], y[idxs]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)

# train GBDT model
model = LGBMClassifier().fit(X_train, y_train)

# fit influence estimator
explainer = BoostIn().fit(model, X_train, y_train)

# estimate training influences on each test instance
influence = explainer.get_local_influence(X_test, y_test)  # shape=(no. train, no. test)

# extract influence values for the first test instance
values = influence[:, 0]  # shape=(no. train,)

# sort training examples from:
# - most positively influential (decreases loss of the test instance the most), to
# - most negatively influential (increases loss of the test instance the most)
training_idxs = np.argsort(values)[::-1]
```

License
---
[Apache License 2.0]()

<!--Reference
---
Brophy, Hammoudeh, and Lowd. [Adapting and Evaluating Influence-Estimation Methods for Gradient-Boosted Decision Trees](). arXiv 2022.

```
```-->