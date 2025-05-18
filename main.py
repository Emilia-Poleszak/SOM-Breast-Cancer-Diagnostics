import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

import som

DATABASE_FILENAME = r'Breast Cancer Coimbra_MLR\dataR2.csv'
EPOCHS = 1000
NO_TRAINING_CALLS = 25
TRAIN_TEST_RATIO = 0.8

def normalise_data(data):
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
    print(y_learn_data)

    return X_learn_data, X_test_data, y_learn_data, y_test_data

def sensitivity_specificity(test: np.array, pred: np.array):
    """
    Calculates sensitivity and specificity of given predicted labels based on true labels.
    In true data: 1 - healthy controls (negative), 2 - patients (positive).
    Prints results in percentage format.
    :param test: True data labels
    :param pred: Predicted data labels
    :return: Percentage sensitivity and specificity
    """
    test_bin = (np.array(test) == 1).astype(int)
    pred_bin = (np.array(pred) == 1).astype(int)

    t_negative, f_positive, f_negative, t_positive = confusion_matrix(test_bin, pred_bin).ravel()

    sensitivity = t_positive / (t_positive + f_negative) * 100
    specificity = t_negative / (t_negative + f_positive) * 100

    print("Sensitivity: {:.2f}%".format(sensitivity))
    print("Specificity: {:.2f}%".format(specificity))

    return sensitivity, specificity

def accuracy(pred: np.array, test: np.array):
    """
    Calculates accuracy of labels predictions: number of predicted labels
    consistent with true labels. Prints results in percentage format.
    :param test: True data labels
    :param pred: Predicted data labels
    :return: Percentage of correct predictions
    """
    a = 0
    for i in range(len(pred)):
        if pred[i] == test[i]:
            a += 1
    a = a / len(pred) * 100
    print("Accuracy: {:.2f}%".format(a))
    return a

def statistics(X_learn, X_test, y_learn, y_test):
    acc, sensitivity, specificity = [], [], []
    som_clf = som.SOMClassifier()
    for i in range(NO_TRAINING_CALLS):
        som_clf.learn(X_learn, y_learn, epochs=EPOCHS)
        y_pred = som_clf.predict(X_test)
        acc.append(accuracy(y_pred, y_test))
        [sens, spec] = sensitivity_specificity(y_test, y_pred)
        sensitivity.append(sens)
        specificity.append(spec)
    print("\nAccuracy:")
    print("Mean: {:.2f}%".format(np.mean(acc)))
    print("Standard Deviation: {:.2f}%".format(np.std(acc)))
    print("\nSensitivity:")
    print("Mean: {:.2f}%".format(np.mean(sensitivity)))
    print("Standard Deviation: {:.2f}%".format(np.std(sensitivity)))
    print("\nSpecificity:")
    print("Mean: {:.2f}%".format(np.mean(specificity)))
    print("Standard Deviation: {:.2f}%".format(np.std(specificity)))

def plot_QE(X_learn, X_test, y_learn, y_test):
    som_clf = som.SOMClassifier()
    errors = som_clf.learn(X_learn, y_learn, epochs=EPOCHS)
    y_pred = som_clf.predict(X_test)
    print("True labels: \n{}".format(np.transpose(y_test)))
    print("Predicted labels: \n{}".format(y_pred))

    accuracy(y_pred, y_test)
    sensitivity_specificity(y_test, y_pred)

    plt.plot([10*i for i in range(len(errors))], errors)
    plt.show()

if __name__ == '__main__':
    X_learn, X_test, y_learn, y_test = load_data()

    ## choose one of the functions below
    statistics(X_learn, X_test, y_learn, y_test)
    # plot_QE(X_learn, X_test, y_learn, y_test)