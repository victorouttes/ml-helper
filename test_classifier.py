import pandas as pd
from sklearn import preprocessing
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
    dataset=dataset,
    tags={'project': 'Testes da lib ml-helper', 'team': 'SOM', 'dataset': 'iris', 'algorithm': 'MLPClassifier'},
    sk_model_class=MLPClassifier,
    sk_preprocessing=[
        preprocessing.StandardScaler,
    ],
).run()
