"""
Functions for constructing and training machine learning models
"""
from sklearn.inspection import permutation_importance

from constants import MODELS


class Model:
    """
    Wrapper class for instantiating and fitting machine learning models. Responsible associating
    models with fits of data splits.
    """
    def __init__(self, model_name=None, model_class=None, model_params=None, dataset=None):
        """
        Constructor for Model objects.

        Parameters
        ----------
        model_name: str
            string name to be used to look up a sklearn estimator
        model_class: str
            sklearn estimator class to be used if directly supplied
        model_params: dict
            kwargs to be passed into sklearn estimator at instantiation
        dataset: DataSet object
            dataset to be trained and evaluated on
        """
        self.model_params = model_params or dict()

        # Instantiate a sklearn estimator
        if model_name:
            self.estimator = MODELS[model_name](**self.model_params)
        elif model_class:
            self.estimator = model_class(**self.model_params)

        # Assign dataset for reference
        self.dataset = dataset

        # Assign data attributes
        self.X_train, self.y_train, self.X_test, self.y_test = self.dataset.split_data()

        # Fit the training data to the model
        self.fit()

    def fit(self, X=None, y=None):
        """
        Fit the model to the data by using the specified feature variables and
        the outcome variable.

        Parameters
        ----------
        X: np.ndarray
            feature matrix to be fit to a sklearn estimator
        y: np.ndarray
            outcome vector to be fit to a sklearn estimator
        """
        X = X or self.X_train
        y = y or self.y_train
        self.estimator.fit(X, y)

    def predict(self, X_pred=None):
        """
        Given a matrix of features, use the trained estimator to create predictions of the outcome vector.

        Paramters
        ---------
        X_pred: np.ndarray
            matrix of feature variable data values to be used to make predictions

        Returns
        -------
        y_pred: np.ndarray
            vector of predicted outcomes created by the trained estimator
        """
        # default to test set if no vector is explicitly supplied
        X_pred = X_pred or self.X_test
        y_pred = self.estimator.predict(X_pred)
        return y_pred

    def get_feature_importances(self, X=None, y=None):
        """
        Compute feature importances used to train a model by using permutation of features.

        Parameters
        ----------
         X: np.ndarray
            feature matrix to be permuted in a sklearn estimator to compute importances
        y: np.ndarray
            outcome vector to be fit to a sklearn estimator for feature importance computation

        Returns
        -------
        sklearn.Bunch object that contains the feature importances, mean, and sd
        """
        X = X or self.X_train
        y = y or self.y_train
        return permutation_importance(self.estimator, X, y)

