import logging

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

from ml_helper.experiment import Experiment
from ml_helper.metrics import regression_metrics


class RegressorExperiment(Experiment):
    def __init__(self, experiment_name: str, sk_model_class, dataset: pd.DataFrame, tags: dict = None, **kwargs):
        self.experiment_name = experiment_name
        self.tags = tags
        self.sk_model_class = sk_model_class
        self.sk_model = None

        self.dataset = dataset

        train, test = train_test_split(dataset)
        self.x_train = train.iloc[:, :-1]
        self.y_train = train.iloc[:, -1].ravel()
        self.x_test = test.iloc[:, :-1]
        self.y_test = test.iloc[:, -1].ravel()

        self.parameters = kwargs

        self.mae = 0.0
        self.mse = 0.0
        self.msle = 0.0
        self.mdae = 0.0
        self.r2 = 0.0

    def _train(self):
        self.sk_model = self.sk_model_class(**self.parameters)
        self.sk_model.fit(self.x_train, self.y_train)

    def _test(self):
        prev = self.sk_model.predict(self.x_test)
        self.mae, self.mse, self.msle, self.mdae, self.r2 = regression_metrics(y_real=self.y_test, y_predict=prev)

    def run(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.experiment_name):
            if self.tags:
                mlflow.set_tags(self.tags)
            self._train()
            self._test()
            for param, val in self.parameters.items():
                mlflow.log_param(param, val)
            mlflow.log_metric('mae', self.mae)
            mlflow.log_metric('mse', self.mse)
            mlflow.log_metric('msle', self.msle)
            mlflow.log_metric('mdae', self.mdae)
            mlflow.log_metric('r2', self.r2)

            signature = infer_signature(self.x_train, self.sk_model.predict(self.x_train))
            mlflow.sklearn.log_model(self.sk_model, 'model', signature=signature)
        logging.info(f'Experiment {self.experiment_name} saved!')

    def __str__(self):
        value = [
            f'Name: {self.experiment_name}',
            f'Regressor: {self.sk_model_class}',
            f'Hyper parameters: {self.parameters}',
            f'MAE: {self.mae}',
            f'MSE: {self.mse}',
            f'MSLE: {self.msle}',
            f'MDAE: {self.mdae}',
            f'R2: {self.r2}',
        ]
        return '\n'.join(value)
