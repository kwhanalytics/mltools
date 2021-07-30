"""
Example usage
python mltools.py --config_path './example/config/config_example.yaml' --iterations 5
"""
import argparse
import yaml

from preprocessing import DataSet
from modeling import Model
from reporting import Report


class MLPipeline:
    """
    MLPipeline encompasses an end-to-end machine learning task with functionality for data ETL, model training,
    and performance evaluation.
    """

    def __init__(self, **kwargs):
        """
        Constructor for MLPipeline.

        Currently supported kwargs
        --------------------------
        baseline: dict
            config for evaluating performance against a baseline model
        dataset: dict
            config for loading, cleaning, transforming, and splitting data
        model: dict
            config for constructing and training the machine learning model
        name: str
            name of the pipeline being run and iteration
        report: str
            config for selecting evaluation metrics and reporting performance
        """
        self.baseline = None
        self.dataset = None
        self.model = None
        self.name = None
        self.report = None

        for key, val in kwargs.items():
            setattr(self, key, val)

    def run(self):
        """Run the MLPipeline end to end, through data preparation, model training, and evaluation."""
        # Create a DataSet object according to config specs
        dataset = DataSet(**self.dataset)

        # Create a Report object according to config specs
        report = Report(report_title=self.name, **self.report)

        # Instantiate, fit, and evaluate a baseline model
        baseline = self.construct_baseline_model(dataset=dataset, **self.baseline)
        report.run(baseline)

        # Instantiate, fit, and evaluate the model
        model = self.construct_model(dataset=dataset, **self.model)
        report.run(model)

    @staticmethod
    def construct_model(name=None, model_params=None, dataset=None):
        """
        Construct a Model object using the specified model parameters and dataset.

        Parameters
        ----------
        name: str
            name of the sklearn estimator to use
        model_params: dict
            kwargs to be passed into sklearn estimator at instantiation
        dataset: DataSet
            dataset to be trained and evaluated on

        Returns
        -------
        Model object representing the main machine learning model of interest
        """
        if name is None:
            raise Exception("No machine learning model specified. Please refer to documentation.")
        return Model(model_name=name, model_params=model_params, dataset=dataset)

    @staticmethod
    def construct_baseline_model(baseline_type=None, baseline_value=None, dataset=None):
        """
        Construct a Model object using the specified model parameters and dataset.

        Parameters
        ----------
        baseline_type: str
            'constant' creates a DummyRegressor, 'model' indicates a separate model to be defined
        baseline_value: float or str
            float constant to be used in DummyRegressor or name of a separate model
        dataset: DataSet
            dataset to be trained and evaluated on

        Returns
        -------
        Model object representing the baseline to be compared against
        """
        if baseline_type is None or baseline_value is None:
            print("No baseline defined, skipping baseline model construction")
        else:
            if baseline_type == "constant":
                baseline_config = {"model_name": "DummyRegressor",
                                   "model_params": {"strategy": "constant", "constant": baseline_value}}
            elif baseline_type == "model":
                baseline_config = {"model_name": baseline_value}
            else:
                raise Exception("Baseline config incorrectly defined. Please refer to documentation.")
            return Model(dataset=dataset, **baseline_config)


def main(config_path, iterations=3):
    with open(config_path, "r") as stream:
        config_file = yaml.safe_load(stream)

    # Create and evaluate individual runs of MLPipeline
    mlp_iterations = dict()
    pipeline_name = config_file["name"]
    for i in range(iterations):
        config_file["name"] = pipeline_name + '_' + str(i)
        mlp = MLPipeline(**config_file)
        mlp.run()
        mlp_iterations[i] = mlp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", help="Specify where the config.yaml file lives.",
                        type=str, required=True)
    parser.add_argument("--iterations", help="Specify the number of iterations you want to run the MLPipeline",
                        type=str, default=3, required=False)
    args = vars(parser.parse_args())
    main(config_path=args['config_path'])
