from sklearn.feature_selection import mutual_info_regression, r_regression
from functools import partial
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.base import BaseEstimator, TransformerMixin


class SelectPercentilePd(SelectPercentile):

    # transform that preserves dataframe structure
    def transform(self, X, y=None):
        X_copy = X.copy()
        transformed_X = super().transform(X_copy)
        sup = self.get_support()
        selected_features = list(X_copy.columns[sup])
        # wind speed is always kept in for binning
        if 'wind speed' not in selected_features:
            selected_features = ['wind speed'] + selected_features
        return X_copy[selected_features]

class SFSPd(sfs):
    # transform that preserves dataframe structure
    def transform(self, X, y=None):
        X_copy = X.copy()
        transformed_X = super().transform(X_copy)
        sup = self.get_support()
        selected_features = list(X_copy.columns[sup])
        # wind speed is always kept in for binning
        if 'wind speed' not in selected_features:
            selected_features = ['wind speed'] + selected_features
        return X_copy[selected_features]



class FeatureSelectionConf:

    selection_algs = [ 'SelectPercentileR',
                       'SelectPercentileMI',
                       'SFS'
    ]

    SelectPercentileMI_hyperparameters = {
       'percentile': [25, 50, 75, 100],
       'score_func': [
           partial(mutual_info_regression, n_neighbors=n) for n in range(3, 10, 3)
        ]
    }

    SelectPercentileR_hyperparameters = {
       'percentile': [25, 50, 75, 100],
       'score_func': [r_regression]
    }

    SFS_hyperparameters = {
       'n_features_to_select': [0.25, 0.50, 0.75, 1.0]
    }


    selectors_and_params = {
        'SelectPercentileR': [SelectPercentilePd, SelectPercentileMI_hyperparameters],
        'SelectPercentileMI': [SelectPercentilePd, SelectPercentileR_hyperparameters]
        #'SFS': [SFSPd, SFS_hyperparameters]
    }
