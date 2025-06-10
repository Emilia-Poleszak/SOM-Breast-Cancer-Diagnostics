import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true, y_pred, positive_class = 2, negative_class = 1):
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
    return np.array([[tn, fp], [fn, tp]])

def show_conf_matrix(conf_matrix: np.ndarray[2,2], lr: float, map_shape: tuple):
    fig, ax = plt.subplots()
    ax.matshow(conf_matrix, cmap='Blues')
    ax.set_xlabel('Predicted', )
    ax.set_ylabel('True')
    ax.text(0, 0, str(conf_matrix[0][0]))
    ax.text(1, 0, str(conf_matrix[0][1]))
    ax.text(0, 1, str(conf_matrix[1][0]))
    ax.text(1, 1, str(conf_matrix[1][1]))
    # ax.set_xticks(["Positive", "Negative"])
    # ax.set_yticks(["Negative", "Positive"])
    ax.set_title(f'lr: {lr}, map shape: {map_shape}')
    fig.show()

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
    return a

def sensitivity_specificity(test: np.array, pred: np.array) -> tuple:
    """
    Calculates sensitivity and specificity of given predicted labels based on true labels.
    In true data: 1 - healthy controls (negative), 2 - patients (positive).
    Prints results in percentage format.
    :param test: True data labels
    :param pred: Predicted data labels
    :return: Percentage sensitivity and specificity
    """

    tp, fp, fn, tn = confusion_matrix(test, pred).ravel()

    sensitivity = tp / (tp + fn) * 100
    specificity = tn / (tn + fp) * 100

    print("Sensitivity: {:.2f}%".format(sensitivity))
    print("Specificity: {:.2f}%\n".format(specificity))
    return sensitivity, specificity