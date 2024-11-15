import numpy as np
from sklearn.ensemble import IsolationForest

class CustomIsolationForest:

    def __init__(self, n_estimators, max_samples, random_state):
        self.clf = IsolationForest(n_estimators=n_estimators,
                                max_samples=max_samples,
                                random_state=random_state)
        self.estimators_ = CustomExtraTreeRegressor()
        
    def fit(self, x):
        self.clf.fit(x)
        self.estimators_ = self.clf.estimators_