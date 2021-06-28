import logging

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from ml_helper.experiment import Experiment
from ml_helper.metrics import classification_metrics


class ClassifierExperiment(Experiment):
    def __init__(self,
                 experiment_name: str,
                 dataset: pd.DataFrame,
                 sk_model_class,
                 sk_preprocessing: list = None,
                 tags: dict = None,
                 **kwargs):
        self.experiment_name = experiment_name
        self.tags = tags
        self.parameters = kwargs
        self.sk_preprocessing = sk_preprocessing
        self.sk_model_class = sk_model_class

        # pipeline definition
        pipeline = []
        if self.sk_preprocessing:
            for preproc in self.sk_preprocessing:
                pipeline.append((f'{preproc.__name__}', preproc()))
        pipeline.append(('', self.sk_model_class(**self.parameters)))
        self.sk_pipeline = Pipeline(pipeline)

        # dataset split
        self.dataset = dataset
        train, test = train_test_split(self.dataset)
        self.x_train = train.iloc[:, :-1]
        self.y_train = train.iloc[:, -1].ravel()
        self.x_test = test.iloc[:, :-1]
        self.y_test = test.iloc[:, -1].ravel()

        # initializing metrics
        self.accuracy = self.precision = self.recall = self.f1 = self.roc_auc = 0.0

    def _train(self):
        self.sk_pipeline.fit(self.x_train, self.y_train)

    def _test(self):
        prev = self.sk_pipeline.predict(self.x_test)
        self.accuracy, self.precision, self.recall, self.f1, self.roc_auc = classification_metrics(
            y_real=self.y_test, y_predict=prev)

    def run(self):
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=self.experiment_name):
            if self.tags:
                mlflow.set_tags(self.tags)
            self._train()
            self._test()
            for param, val in self.parameters.items():
                mlflow.log_param(param, val)
            mlflow.log_metric('accuracy', self.accuracy)
            mlflow.log_metric('precision', self.precision)
            mlflow.log_metric('recall', self.recall)
            mlflow.log_metric('f1', self.f1)
            mlflow.log_metric('roc_auc', self.roc_auc)

            signature = infer_signature(self.x_train, self.sk_pipeline.predict(self.x_train))
            mlflow.sklearn.log_model(self.sk_pipeline, 'pipeline', signature=signature)
        logging.info(f'Experiment {self.experiment_name} saved!')

    def __str__(self):
        value = [
            f'Name: {self.experiment_name}',
            f'Classifier: {self.sk_model_class}',
            f'Hyper parameters: {self.parameters}',
            f'Accuracy: {self.accuracy}',
            f'Precision: {self.precision}',
            f'Recall: {self.recall}',
            f'F1: {self.f1}',
            f'ROC AUC: {self.roc_auc}',
        ]
        return '\n'.join(value)
