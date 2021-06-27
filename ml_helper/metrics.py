from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


def classification_metrics(y_real, y_predict):
    average = 'micro'
    accuracy = accuracy_score(y_real, y_predict)
    precision = precision_score(y_real, y_predict, average=average)
    recall = recall_score(y_real, y_predict, average=average)
    f1 = f1_score(y_real, y_predict, average=average)
    # run LabelBinazirer, as ROC is a one class metric
    label_binarizer = preprocessing.LabelBinarizer()
    y_real_b = label_binarizer.fit_transform(y_real)
    y_predict_b = label_binarizer.transform(y_predict)
    roc_auc = roc_auc_score(y_real_b, y_predict_b, multi_class='ovr')
    return accuracy, precision, recall, f1, roc_auc


def regression_metrics(y_real, y_predict):
    mae = mean_absolute_error(y_real, y_predict)
    mse = mean_squared_error(y_real, y_predict)
    msle = mean_squared_log_error(y_real, y_predict)
    mdae = median_absolute_error(y_real, y_predict)
    r2 = r2_score(y_real, y_predict)
    return mae, mse, msle, mdae, r2
