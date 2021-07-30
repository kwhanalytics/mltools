"""
Functions for loading, cleaning, and transforming data features and outcome variables.
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from constants import SCALERS


class DataSet:
    """
    Class responsible for data loading, preprocessing, and splitting.
    """
    def __init__(self, data_file=None, cleaning=None, scaling=None, splitting=None):
        """
        Constructor for DataSet objects.

        Parameters
        ----------
        data_file: dict
            config from yaml related to loading data
        cleaning: dict
            config from yaml related to cleaning data
        scaling: dict
            config from yaml related to scaling data values
        splitting: dict
            config from yaml related to splitting data into training and test sets
        """
        self.filepath = None
        self.use_cols = None
        self.outcome = None
        self.primary_key = None
        self.categorical_cols = None
        for key, val in data_file.items():
            setattr(self, key, val)

        self.cleaning = cleaning or list()

        self.outcome_scaler = None
        self.feature_scaler = None
        self.features_to_scale = None
        if scaling:
            for key, val in scaling.items():
                setattr(self, key, val)
            if self.outcome_scaler:
                self.outcome_scaler = SCALERS[self.outcome_scaler]()

        self.split_method = None
        self.split_params = None
        if splitting:
            for key, val in splitting.items():
                setattr(self, key, val)

        self.data = self.load_data()
        self.preprocess_data()
        self.features = self.get_features()

    def load_data(self):
        """Read in csv formatted data as a pd.DataFrame"""
        data = pd.read_csv(self.filepath)
        return data

    def get_features(self):
        """Get the columns of the dataset that will be used as features to train the model"""
        not_features = [col for col in self.data.columns
                        if col not in self.use_cols
                        and col not in self.categorical_cols]
        not_features.extend([self.primary_key, self.outcome])
        return self.data.drop(not_features, axis=1).columns

    def preprocess_data(self):
        """Run preprocessing steps on data according to config yaml"""
        for step in self.cleaning:
            getattr(self, step)()

    def split_data(self, make_validation_set=False):
        """
        Split the data into training and test sets and then into features matrix and outcome vector.

        Parameters
        -----------
        make_validation_set: bool
            if True, run train_test_split twice, where the first call sets aside a validation set

        Returns
        -------
        X_train, y_train, X_test, y_test
        Optionally X_valid, y_valid
        """
        if make_validation_set:
            training_data, test_data, validation_data = self.split_training_and_test(make_validation_set)
            X_train, y_train = self.split_features_and_outcome(training_data)
            X_test, y_test = self.split_features_and_outcome(test_data)
            X_valid, y_valid = self.split_features_and_outcome(validation_data)
            return X_train, y_train, X_test, y_test, X_valid, y_valid
        else:
            training_data, test_data, validation_data = self.split_training_and_test(make_validation_set)
            X_train, y_train = self.split_features_and_outcome(training_data)
            X_test, y_test = self.split_features_and_outcome(test_data)
            return X_train, y_train, X_test, y_test

    def scale_outcome(self):
        """Fit a scaler object and transform the outcome vector"""
        outcome_array = self.data.loc[:, self.outcome].values
        return self.outcome_scaler.fit_transform(outcome_array.reshape(-1, 1))

    def inverse_scale_outcome(self, outcome_array):
        """Using a fit scaler object, inverse transform the outcome vector for reporting interpretability"""
        return self.outcome_scaler.inverse_transform(outcome_array)

    def split_training_and_test(self, split_kwargs=None, make_validation_set=True):
        """
        Split a DataFrame into training and test subsets using a random seed.
        Optionally set aside a validation set.

        Paramters
        ----------
        data: pd.DataFrame
            matrix of data including primary key, predictor features, and outcome variable
        split_kwargs: dict
            dictionary of supported kwargs for sklearn.model_selection.train_test_split
        make_validation_set: bool
            if True, run train_test_split twice, where the first call sets aside a validation set

        Returns
        --------
        training_data, test_data, validation_data (if specified)
        """
        split_kwargs = split_kwargs or dict()
        if make_validation_set:
            training_data, validation_data = train_test_split(self.data, **split_kwargs)
            training_data, test_data = train_test_split(training_data, **split_kwargs)
            return training_data, test_data, validation_data
        else:
            training_data, test_data = train_test_split(self.data, **split_kwargs)
            return training_data, test_data

    def split_features_and_outcome(self, data, arrays_only=True):
        """
        Split a DataFrame into X and y arrays according to specified feature and outcome variable names.

        Parameters
        ----------
        data: pd.DataFrame
            pandas dataframe with features and outcome variable names in the columns and samples in the index
        arrays_only: bool
            if True, extract the underlying np.ndarrays and remove pd.DataFrame features like columns and names

        Returns
        -------
        X: np.ndarray
            matrix of data values of the subset of features
        y: np.ndarray
            vector of data values of the
        """
        # drop primary key if it exists in data as a column
        if self.primary_key in data.columns:
            data = data.drop(self.primary_key, axis=1)

        X = data.loc[:, self.features]
        y = data.loc[:, self.outcome]
        if arrays_only:
            X = X.values
            y = y.values
        return X, y

    def keep_first_duplicate(self):
        """Drop the second occurance of a duplicate row"""
        duplicate_index = self.data[self.data.duplicated()].index
        return self.data.drop(duplicate_index)

    def drop_outcome_outliers(self):
        """Drop rows of data where the outcome value is an outlier"""
        def get_outliers(data, m=5):
            """Using the distribution of a continuous variable, identify outliers and their standardized distance from
            the mean"""
            d = np.abs(data.values - np.median(data.values))
            mdev = np.median(d)
            s = d / mdev if mdev else 0.
            # returns a series bool of where the outliers are
            outliers = data[s > m]
            return outliers

        outcome_outlier_index = get_outliers(self.data.loc[:, self.outcome]).index
        return self.data.drop(outcome_outlier_index)

    def drop_missing_values(self):
        """Drop any rows of data that have missing values"""
        missing_index = self.data[self.data.isna().any(axis=1)].index
        return self.data.drop(missing_index)

    def one_hot_categorical_variables(self):
        """Transform categorical variables into one-hot dummy variables for machine learning tasks"""
        self.data = pd.get_dummies(self.data, prefix=self.categorical_cols, columns=self.categorical_cols)
