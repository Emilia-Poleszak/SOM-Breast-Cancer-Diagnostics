import numpy as np
import pandas as pd

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
    n = np.count_nonzero(y==1)
    X1 = X.iloc[:n]
    X2 = X.iloc[n:]
    y1 = y.iloc[:n]
    y2 = y.iloc[n:]

    # splitting X and y: 80% for training and 20% for tests
    rows_1 = int(0.8 * n)
    rows_2 = int(0.8 * (len(X) - n))

    X_learn_data = pd.concat([X1.iloc[:rows_1], X2.iloc[:rows_2]], axis=0)
    X_test_data = pd.concat([X1.iloc[rows_1:], X2.iloc[rows_2:]], axis=0)
    y_learn_data = pd.concat([y1.iloc[:rows_1], y2.iloc[:rows_2]], axis=0)
    y_test_data = pd.concat([y1.iloc[rows_1:], y2.iloc[rows_2:]], axis=0)

    return X_learn_data.to_numpy(), X_test_data.to_numpy(), y_learn_data.to_numpy(), y_test_data.to_numpy()

if __name__ == '__main__':
    X_learn, X_test, y_learn, y_test = load_data()

    som_clf = som.SOMClassifier()
    som_clf.learn(X_learn, y_learn)
    y_pred = som_clf.predict(X_test)
    print(np.transpose(y_test))
    print(y_pred)
    # accuracy = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] == y_test[i]:
    #         accuracy += 1
    # accuracy = accuracy / len(y_pred)