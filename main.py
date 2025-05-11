import numpy as np
import pandas as pd

import som

def load_data():
    data = pd.read_csv(r'Breast Cancer Coimbra_MLR\dataR2.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # splitting data by classes and statistical analysis
    n = np.count_nonzero(y==1)
    X1 = X.iloc[:n].to_numpy()
    X2 = X.iloc[n:].to_numpy()
    y1 = y.iloc[:n].to_numpy()
    y2 = y.iloc[n:].to_numpy()
    return X.to_numpy(), y.to_numpy()-1

if __name__ == '__main__':
    X, y = load_data()

    X_learn = X[int(0.8*len(X)):]
    X_test = X[:int(0.8*len(X))]
    y_learn = y[int(0.8*len(X)):]
    y_test = y[:int(0.8*len(X))]

    som_clf = som.SOMClassifier()
    som_clf.learn(X_learn, y_learn)
    y_pred = som_clf.predict(X_test)
    print(y_pred)
    # accuracy = 0
    # for i in range(len(y_pred)):
    #     if y_pred[i] == y_test[i]:
    #         accuracy += 1
    # accuracy = accuracy / len(y_pred)