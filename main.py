import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import som, metrics

DATABASE_FILENAME = r'Breast Cancer Coimbra_MLR\dataR2.csv'
EPOCHS = 3000
NO_TRAINING_CALLS = 25
TRAIN_TEST_RATIO = 0.75

TESTED_LRS = [0.001, 0.01, 0.2, 0.8]
TESTED_MAP_SIZES = [(3,3), (7,7), (12,12), (25,25)]

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

def statistics(learn_x: np.ndarray, test_x: np.ndarray,
               learn_y: np.ndarray, test_y: np.ndarray):
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
        acc.append(metrics.accuracy(pred_y, test_y))
        [sens, spec] = metrics.sensitivity_specificity(test_y, pred_y)
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

def single_training(x_learn_data: np.ndarray, x_test_data: np.ndarray,
                    y_learn_data: np.ndarray, y_test_data: np.ndarray,
                    learning_rate: float = 0.2, std: float = 1.0,
                    map_shape: tuple = (7, 7)):
    """
    Calculates and prints accuracy, sensitivity and specificity of SOM.
    Creates a graph of quantisation error.
    :param x_learn_data: Learning data X
    :param x_test_data: Testing data X
    :param y_learn_data: Learning data y
    :param y_test_data: Testing data y
    :param learning_rate: Learning rate
    :param std: Standard deviation
    :param map_shape: Shape of map
    """
    som_clf = som.SOMClassifier(map_shape=map_shape)
    errors = som_clf.train(x_learn_data, y_learn_data, x_test_data, y_test_data,
                           epochs=EPOCHS, learning_rate=learning_rate, std=std)
    y_pred = som_clf.predict(x_test_data)

    print(f"\nlr: {learning_rate}, map shape: {map_shape}")
    print("True labels: \n{}".format(np.transpose(y_test_data)))
    print("Predicted labels: \n{}".format(y_pred))

    conf_mat = metrics.confusion_matrix(y_test_data, y_pred)
    metrics.show_conf_matrix(conf_mat, learning_rate, map_shape)

    acc = metrics.accuracy(y_pred, y_test_data)
    print("Accuracy: {:.2f}%".format(acc))
    metrics.sensitivity_specificity(y_test_data, y_pred)

    fig, ax = plt.subplots()
    ax.plot([10 * i for i in range(len(errors))], errors)
    ax.set_title(f"lr: {learning_rate}, map shape: {map_shape}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Quantization Error")
    fig.show()

def test_parameters(x_learn_data: np.ndarray, x_test_data: np.ndarray,
                    y_learn_data: np.ndarray, y_test_data: np.ndarray):
    for lr in TESTED_LRS:
        for map_shape in TESTED_MAP_SIZES:
            single_training(x_learn_data, x_test_data, y_learn_data, y_test_data,
                            learning_rate=lr, map_shape=map_shape)


if __name__ == '__main__':
    X_learn, X_test, y_learn, y_test = load_data()

    ## choose one of the functions below
    # statistics(X_learn, X_test, y_learn, y_test)
    single_training(X_learn, X_test, y_learn, y_test)

    #test_parameters(X_learn, X_test, y_learn, y_test)

    plt.show()
