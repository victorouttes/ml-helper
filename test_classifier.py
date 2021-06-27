import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from ml_helper.classifier import ClassifierExperiment

dataset = pd.read_csv('/home/victor/Downloads/iris.data')
dataset.columns = [
    'sepal length in cm',
    'sepal width in cm',
    'petal length in cm',
    'petal width in cm',
    'class',
]

ClassifierExperiment(
    experiment_name='ML HELPER',
    tags={'project': 'Testes da lib ml-helper', 'team': 'SOM', 'dataset': 'iris', 'algorithm': 'SVC'},
    sk_model_class=SVC,
    dataset=dataset,
    decision_function_shape='ovr'
).run()

ClassifierExperiment(
    experiment_name='ML HELPER',
    tags={'project': 'Testes da lib ml-helper', 'team': 'SOM', 'dataset': 'iris', 'algorithm': 'MLPClassifier'},
    sk_model_class=MLPClassifier,
    dataset=dataset,
).run()
