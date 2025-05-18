import numpy as np

class SOMClassifier(object):
    """
    Implementation of Self-Organizing Map
    """
    def __init__(self, map_shape: (int, int) = (7,7)):
        self.map_shape = map_shape
        self.w_map = np.array([])
        self.neuron_labels = {}

    def train(self, x_train, y_train, x_val = None, y_val = None,
              epochs: int = 1000,
              learning_rate: float = 0.2,
              std: float = 1.0):
        """
        Organises neuron map based on given data.
        :param x_train: Input dataset for training
        :param y_train: Output dataset for training
        :param x_val: Input dataset for validation
        :param y_val: Output dataset for validation
        :param epochs: Number of epochs
        :param learning_rate: Learning rate used for updating weights
        :param std: Standard deviation for gaussian neighborhood function
        :return: Quantization errors evolution during training
        """
        self.w_map = np.random.rand(self.map_shape[0], self.map_shape[1], x_train.shape[1])
        lr = learning_rate
        s = std
        qe = []
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train

        for epoch in range(epochs):
            xk = x_train[np.random.randint(0, len(x_train))]
            bmu_idx = self.find_bmu(xk)
            h = self.h_func(bmu_idx, s)

            for i in range(self.map_shape[0]):
                for j in range(self.map_shape[1]):
                    self.w_map[i,j] += lr * h[i,j] * (xk - self.w_map[i,j])

            if epoch % 10 == 0:
                errors = [np.linalg.norm(x - self.w_map[self.find_bmu(x)]) for x in x_val]
                qe.append(np.mean(errors))

            lr = learning_rate * (1 - epoch/epochs)
            s = std * (1 - epoch/epochs)

        self.set_labels(x_train, y_train)
        return np.array(qe)

    def predict(self, samples):
        """
        Predicts a label of given data samples.
        :param samples: Data samples
        :return: Data labels
        """
        results = []
        for sample in samples:
            bmu_idx = self.find_bmu(sample)
            results.append(self.neuron_labels[bmu_idx])
        return np.array(results)

    def find_bmu(self, xk) -> tuple[int, int]:
        """
        Finds best matching unit's coordinates based on Euclidean distance
        between a neuron and a data sample.
        :param xk: Data sample
        :return: (x, y) coordinates of best matching unit in weights map
        """
        winner = (0, 0)
        shortest_distance = np.inf
        for row in range(self.map_shape[0]):
            for col in range(self.map_shape[1]):
                distance = self.e_distance(self.w_map[row][col], xk)
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = (row, col)
        return winner

    def h_func(self, bmu_idx: tuple[int, int], std: float):
        """
        Generates gaussian neighborhood function value for each neuron in self.w_map.
        :param bmu_idx: Best matching unit's index
        :param std: Standard deviation for gaussian neighborhood function
        :return: Matrix of gaussian neighborhood function values
        """
        h = np.empty(self.map_shape)
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                distance = SOMClassifier.e_distance(np.array(bmu_idx), np.array([i, j]))
                h[i, j] = np.exp(-(distance ** 2) / (2 * (std ** 2)))
        return h

    def set_labels(self, x_train, y_train):
        """
        Creates labels of trained map based on training data and it's labels.
        :param x_train:
        :param y_train:
        :return:
        """
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                distances = [SOMClassifier.e_distance(self.w_map[i, j], x) for x in x_train]
                self.neuron_labels[(i, j)] = y_train[np.argmin(distances)]

    @staticmethod
    def e_distance(a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculates Euclidean distance between two vectors.
        :param a: First vector
        :param b: Second vector
        :return: Euclidean distance
        """
        return np.sqrt(np.sum((a - b) ** 2))