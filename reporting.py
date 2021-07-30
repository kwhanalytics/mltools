"""
Functions for evaluating machine learning models and reporting performance.
"""
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer

from constants import DEFAULT_METRICS


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


METRICS = {
    "rmse": rmse,
    "r2": r2_score
}


class Report:
    """
    Class responsible for evaluating model performance and reporting on scores.
    """
    def __init__(self, report_dir, report_title, metrics=None, write_to_csv=True):
        """
        Constructor for Report objects.

        Parameters
        ----------
        report_dir: str
            directory for where to write any reports
        report_title: str
            title of the report, combination of pipeline name and iteration value
        metrics: list
            list of metrics to be used when scoring model performance
        write_to_csv: bool
            if True, write all reports to csv files in report_dir, else print to stdout
        """
        self.report_dir = report_dir
        self.report_title = report_title
        self.metrics = metrics or DEFAULT_METRICS
        self.write_to_csv = write_to_csv

    def summarize_outcome_variable(self, model):
        """
        Statistically describe the outcome variable used in machine learning evaluation.

        Parameters
        ----------
        model: Model
            Model with corresponding DataSet attribute
        """
        outcome_var_summary = model.dataset.data.loc[:, model.dataset.outcome].describe()
        if self.write_to_csv:
            filepath = f"{self.report_dir}/{self.report_title}_outcome_var_summary.csv"
            outcome_var_summary.to_csv(filepath)
        else:
            print(outcome_var_summary)

    def evaluate(self, model, cv=True, cv_params=None, X_eval=None):
        """
        Score a model using its predict function

        Parameters
        ----------
        model: Model
            Model with corresponding DataSet attribute
        cv: bool
            if true, cross validate
        cv_params: dict
            kwargs to be passed into cross_validate function

        Returns
        -------
        scores: dict
            dictionary of selected metrics and corresponding scores
        """
        if cv:
            if cv_params is None and self.metrics:
                cv_params = {"scoring": {metric_name: make_scorer(METRICS[metric_name]) for metric_name in self.metrics}}
            else:
                cv_params = cv_params or dict()
            scores = cross_validate(model.estimator, model.X_train, model.y_train, **cv_params)
        elif X_eval:
            y_pred = model.predict(X_eval)
            scores = {metric_name: METRICS[metric_name](model.y, y_pred) for metric_name in self.metrics}
        else:
            raise Exception("Must either specify cross validation or provide explicit evalution set to Report.evalute")
        return scores

    def report_scores(self, scores):
        """
        Report scores in a pd.DataFrame format.

        Parameters
        ----------
        scores: dict
            dictionary of selected metrics and corresponding scores
        """
        # convert scores to pd.DataFrame for writing to csv
        scores = pd.DataFrame(scores)
        if self.write_to_csv:
            report_filepath = f"{self.report_dir}/{self.report_title}_scores.csv"
            scores.to_csv(report_filepath)
        # print to stdout
        else:
            print(scores)

    def report_feature_importances(self, model):
        """
        Report feature importances, either from an inherent model attribute (i.e. Random Forest)
        or from feature importance permutation.

        Parameters
        ----------
        model: Model
            Model with corresponding DataSet attribute
        """
        if hasattr(model.estimator, 'feature_importances_'):
            feature_importances = model.estimator.feature_importances_
        else:
            feature_importances = model.get_feature_importances()

        # match the features with their columns
        feature_names = model.dataset.data.drop([model.dataset.primary_key, model.dataset.outcome], axis=1).columns
        final_feature_importances = pd.Series({name: val for name, val in zip(feature_names, feature_importances)},
                                             name="importance").sort_values(ascending=False)
        if self.write_to_csv:
            report_filepath = f"{self.report_dir}/{self.report_title}_feature_importances.csv"
            final_feature_importances.to_csv(report_filepath)
        else:
            print(final_feature_importances)

    def run(self, model):
        self.summarize_outcome_variable(model)
        scores = self.evaluate(model)
        self.report_feature_importances(model)
        self.report_scores(scores)