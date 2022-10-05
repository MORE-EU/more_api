from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from modules.preprocessing import split_to_bins
from copy import deepcopy
import numpy as np

class DirectYawEstimator(RegressorMixin, BaseEstimator):

    def __init__(self, base_estimator,
                 bin_size=1,
                 min_speed=5,
                 max_speed=11):
        self.base_estimator = base_estimator
        self.base_estimators = []
        self.bin_size=bin_size
        self.min_speed=min_speed
        self.max_speed=max_speed


    def fit(self, X, y):
        X = X.copy()
        y = y.copy()
        self._bin_data(X)
        self._custom_fit(X, y)
        return self

    def predict(self, X):
        X = X.copy()
        self._bin_data(X)
        return self._custom_predict(X)


    def _bin_data(self, X, bin_feature='wind speed'):
        self._bin_masks = split_to_bins(X.copy(), self.bin_size, self.min_speed,
                                        self.max_speed, bin_feature)

    def _custom_fit(self, X, y):
        binned_data = []
        for b in self._bin_masks:
            X_temp = X[b]
            y_temp = y[b]
            binned_data.append((X_temp.copy(), y_temp.copy()))

        for bin_n, (X_t, y_t) in enumerate(binned_data):
            if X_t.shape[0] >= 100:
                est = clone(self.base_estimator)
                model = est.fit(X_t, y_t)
                self.base_estimators.append(deepcopy(model))
            else:
                self.base_estimators.append(None)

    def _custom_predict(self, X):

        binned_data = []
        all_preds = {}
        for b in self._bin_masks:
            X_temp = X[b]
            binned_data.append(X_temp.copy())


        for bin_n, X_t in enumerate(binned_data):
            dataset_preds = []
            if X_t.shape[0] >= 1 and self.base_estimators[bin_n] is not None:
                est = self.base_estimators[bin_n]
                test_preds = est.predict(X_t)
                dataset_preds.append(test_preds)
            else:
                pass
            if dataset_preds != []:
                all_preds[bin_n] = np.mean(dataset_preds, axis=0)
            else:
                all_preds[bin_n] = None


        X['y_pred'] = np.nan
        for bin_n, b in enumerate(self._bin_masks):
            if all_preds[bin_n] is not None:
                predictions = all_preds[bin_n]
                X.loc[b, 'y_pred'] = predictions
            else:
                pass
        prediction = np.abs(X['y_pred'])

        return self._post_process(prediction)

    @staticmethod
    def _post_process(prediction):
        prediction = prediction.interpolate().bfill()
        return prediction
