import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import som2 as som

DATABASE_FILENAME = r'Breast Cancer Coimbra_MLR\dataR2.csv'
EPOCHS = 1000
NO_TRAINING_CALLS = 25
TRAIN_TEST_RATIO = 0.7

def normalise_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalises data to [0,1] by calculating max and min value of each column.
    :param data: Raw data
    :return: Normalised data
    """
    normalised = data.copy()

    for i in data.select_dtypes(include='number').columns:
        min_val = data[i].min()
        max_val = data[i].max()
        if min_val == max_val:
            normalised[i] = 0.0
        else:
            normalised[i] = (data[i] - min_val) / (max_val - min_val)

    return normalised


def load_data():
    """
    Loads data and splits it for X and y, by classes and for training and test.
    :return: Numpy arrays: X for training, y for training, X for tests, y for tests
    """
    data = pd.read_csv(DATABASE_FILENAME)

    X = normalise_data(data.iloc[:, :-1])
    y = data.iloc[:, -1]

    # splitting data by classes
    n = np.count_nonzero(y == 1)
    X1 = X.iloc[:n]
    X2 = X.iloc[n:]
    y1 = y.iloc[:n]
    y2 = y.iloc[n:]

    # splitting X and y: 80% for training and 20% for tests
    rows_1 = int(TRAIN_TEST_RATIO * n)
    rows_2 = int(TRAIN_TEST_RATIO * (len(X) - n))

    X_learn_data = pd.concat([X1.iloc[:rows_1], X2.iloc[:rows_2]], axis=0).to_numpy()
    X_test_data = pd.concat([X1.iloc[rows_1:], X2.iloc[rows_2:]], axis=0).to_numpy()
    y_learn_data = pd.concat([y1.iloc[:rows_1], y2.iloc[:rows_2]], axis=0).to_numpy()
    y_test_data = pd.concat([y1.iloc[rows_1:], y2.iloc[rows_2:]], axis=0).to_numpy()

    return X_learn_data, X_test_data, y_learn_data, y_test_data


X_learn, X_test, y_learn, y_test = load_data()

clf = som.SOMClassifier()
accuracy = []
for i in range(NO_TRAINING_CALLS):
    clf.train(X_learn, y_learn, epochs=EPOCHS)
    y_pred = clf.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_pred)
    print(i, ": ", acc)
    accuracy.append(acc)
print("Mean accuracy: ", np.mean(accuracy))
print("Standard deviation: ", np.std(accuracy))