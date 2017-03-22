from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import numpy as np


def eval_linear_model(X, y, features, train_indices, test_indices, fit_intercept, normalize):
    lm = LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
    lm.fit(X[train_indices, features], y)
    predictions = lm.predict(X[test_indices, features])
    mse = np.mean((predictions - y[test_indices]) ** 2)
    return mse


class StepwiseLinearRegression(object):
    def __init__(self, percent_gain=0.02, num_folds=5, fit_intercept=True, normalize=False, n_jobs=1):
        self.percent_gain = percent_gain
        self.num_folds = num_folds
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.linear_model = LinearRegression(fit_intercept=self.fit_intercept, normalize=self.normalize)
        self.features = []

    def fit(self, X, y):
        current_percent_gain = 1
        current_score = 100000
        feature_indices = list(range(X.shape[1]))
        with Parallel(n_jobs=self.n_jobs) as parallel:
            while current_percent_gain > self.percent_gain and len(feature_indices) > 0:
                all_scores = np.zeros((len(feature_indices), self.num_folds))
                kf = KFold(n_splits=self.num_folds)
                for k in range(self.num_folds):
                    train_indices, test_indices = kf.split(X)
                    all_scores[:, k] = parallel(delayed(eval_linear_model)(X,
                                                                           y,
                                                                           self.features + [f],
                                                                           train_indices,
                                                                           test_indices,
                                                                           self.fit_intercept,
                                                                           self.normalize,
                                                                           ) for f in feature_indices)
                best_model = np.argmin(all_scores.mean(axis=1))
                best_score = all_scores.mean(axis=1).min()
                current_percent_gain = (current_score - best_score) / current_score
                if current_percent_gain > self.percent_gain:
                    current_score = best_score
                    self.features.append(feature_indices[best_model])
                    feature_indices.remove(feature_indices[best_model])
                    self.linear_model.fit(X[:, self.features], y)

    def predict(self, X):
        return self.linear_model.predict(X[:, self.features])
