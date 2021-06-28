import mlflow
logged_model = 's3://mlflow/4/44e7f04e179c452bb56c68532716fdd6/artifacts/pipeline'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd
data = [
    [5., 5., 2., 7.8]
]
prev = loaded_model.predict(pd.DataFrame(data, columns=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']))
print(prev)
