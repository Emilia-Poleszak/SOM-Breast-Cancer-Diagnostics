import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

import som


def load_data():
    """
    Loads data and splits it for X and y, by classes and for training and test.
    :return: Numpy arrays: X for training, y for training, X for tests, y for tests
    """
    data = pd.read_csv(r'Breast Cancer Coimbra_MLR\dataR2.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # splitting data by classes
    n = np.count_nonzero(y == 1)
    X1 = X.iloc[:n]
    X2 = X.iloc[n:]
    y1 = y.iloc[:n]
    y2 = y.iloc[n:]

    # splitting X and y: 80% for training and 20% for tests
    rows_1 = int(0.8 * n)
    rows_2 = int(0.8 * (len(X) - n))

    X_learn_data = pd.concat([X1.iloc[:rows_1], X2.iloc[:rows_2]], axis=0).to_numpy()
    X_test_data = pd.concat([X1.iloc[rows_1:], X2.iloc[rows_2:]], axis=0).to_numpy()
    y_learn_data = pd.concat([y1.iloc[:rows_1], y2.iloc[:rows_2]], axis=0)
    y_test_data = pd.concat([y1.iloc[rows_1:], y2.iloc[rows_2:]], axis=0)

    # normalising data to [0,1]
    scaler = MinMaxScaler()
    X_learn_data = scaler.fit_transform(X_learn_data)
    X_test_data = scaler.transform(X_test_data)

    # X_learn_data, y_learn_data = shuffle(X_learn_data, y_learn_data, random_state=42)
    # X_test_data, y_test_data = shuffle(X_test_data, y_test_data, random_state=42)

    return X_learn_data, X_test_data, y_learn_data.to_numpy(), y_test_data.to_numpy()


def sensitivity_specificity(test: np.array, pred: np.array):
    """
    Calculates sensitivity and specificity of given predicted labels based on true labels.
    Prints results in percentage format.
    :param test: True data labels
    :param pred: Predicted data labels
    """
    test_bin = (np.array(test) == 2).astype(int)
    pred_bin = (np.array(pred) == 2).astype(int)

    t_negative, f_positive, f_negative, t_positive = confusion_matrix(test_bin, pred_bin).ravel()

    sensitivity = t_positive / (t_positive + f_negative) * 100
    specificity = t_negative / (t_negative + f_positive) * 100

    print("Sensitivity: {:.2f}%".format(sensitivity))
    print("Specificity: {:.2f}%".format(specificity))


def accuracy(pred: np.array, test: np.array):
    """
    Calculates accuracy of labels predictions: number of predicted labels
    consistent with true labels. Prints results in percentage format.
    :param test: True data labels
    :param pred: Predicted data labels
    """
    a = 0
    for i in range(len(pred)):
        if pred[i] == test[i]:
            a += 1
    a = a / len(pred) * 100
    print("Accuracy: {:.2f}%".format(a))


if __name__ == '__main__':
    X_learn, X_test, y_learn, y_test = load_data()

    som_clf = som.SOMClassifier()
    som_clf.learn(X_learn, y_learn, epochs=2000)
    y_pred = som_clf.predict(X_test)
    print("True labels: \n{}".format(np.transpose(y_test)))
    print("Predicted labels: \n{}".format(y_pred))

    accuracy(y_pred, y_test)
    sensitivity_specificity(y_test, y_pred)
