import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from ml_helper.regressor import RegressorExperiment

dataset = pd.read_csv('/home/victor/Downloads/archive/Boston.csv')
dataset = dataset.iloc[:, 1:]

RegressorExperiment(
    experiment_name='ML HELPER',
    tags={'project': 'Testes da lib ml-helper', 'team': 'SOM', 'dataset': 'boston', 'algorithm': 'RandomForestRegressor'},
    sk_model_class=RandomForestRegressor,
    dataset=dataset,
).run()

RegressorExperiment(
    experiment_name='ML HELPER',
    tags={'project': 'Testes da lib ml-helper', 'team': 'SOM', 'dataset': 'boston', 'algorithm': 'GradientBoostingRegressor'},
    sk_model_class=GradientBoostingRegressor,
    dataset=dataset,
).run()
