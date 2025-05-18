import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import som

DATABASE_FILENAME = r'Breast Cancer Coimbra_MLR\dataR2.csv'
EPOCHS = 1000
NO_TRAINING_CALLS = 25
TRAIN_TEST_RATIO = 0.75


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


def confusion_matrix(y_true, y_pred, positive_class=2, negative_class=1):
    """
    Calculates confusion matrix for given true and predicted values.
    :param y_true: True class labels
    :param y_pred: Predicted class labels
    :param positive_class: Positive class label
    :param negative_class: Negative class label
    :return:
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length.")

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            if y_pred[i] == positive_class:
                tp += 1
            elif y_pred[i] == negative_class:
                tn += 1
        else:
            if y_pred[i] == positive_class:
                fp += 1
            if y_pred[i] == negative_class:
                fn += 1

    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100

    print("Sensitivity: {:.2f}%".format(sensitivity))
    print("Specificity: {:.2f}%\n".format(specificity))

    return sensitivity, specificity


def accuracy(pred: np.array, test: np.array) -> float:
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


def statistics(learn_x: np.array, test_x: np.array, learn_y: np.array, test_y: np.array):
    """
    Calculates and prints statistics: mean accuracy, sensitivity and specificity of SOM.
    :param learn_x: Learning data X
    :param test_x: Testing data X
    :param learn_y: Learning data y
    :param test_y: Testing data y
    """
    acc, sensitivity, specificity = [], [], []
    som_clf = som.SOMClassifier()

    for i in range(NO_TRAINING_CALLS):
        som_clf.train(learn_x, learn_y, test_x, test_y, epochs=EPOCHS)
        pred_y = som_clf.predict(test_x)
        acc.append(accuracy(pred_y, test_y))
        [sens, spec] = confusion_matrix(test_y, pred_y)
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


def plot_qe(x_learn_data: np.array, x_test_data: np.array, y_learn_data: np.array, y_test_data: np.array):
    """
    Calculates and prints accuracy, sensitivity and specificity of SOM.
    Creates a graph of quantisation error.
    :param x_learn_data: Learning data X
    :param x_test_data: Testing data X
    :param y_learn_data: Learning data y
    :param y_test_data: Testing data y
    """
    som_clf = som.SOMClassifier()
    errors = som_clf.train(x_learn_data, y_learn_data, x_test_data, y_test_data, epochs=EPOCHS)
    y_pred = som_clf.predict(x_test_data)

    print("True labels: \n{}".format(np.transpose(y_test_data)))
    print("Predicted labels: \n{}".format(y_pred))

    accuracy(y_pred, y_test_data)
    confusion_matrix(y_test_data, y_pred)

    fig, ax = plt.subplots()
    ax.plot([10 * i for i in range(len(errors))], errors)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Quantization Error")
    fig.show()


if __name__ == '__main__':
    X_learn, X_test, y_learn, y_test = load_data()

    ## choose one of the functions below
    statistics(X_learn, X_test, y_learn, y_test)
    # plot_qe(X_learn, X_test, y_learn, y_test)

    plt.show()
