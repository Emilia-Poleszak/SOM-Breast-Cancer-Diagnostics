import numpy as np

class SOMClassifier():
    """
    Implementation of Self-Organizing Map
    """
    def __init__(self, map_shape: (int, int) = (6,6)):
        self.map_shape = map_shape
        self.w_map = np.array([])
        self.neuron_labels = {}

    def train(self, x_train, y_train,
              epochs: int = 1000,
              learning_rate: float = 0.2,
              std: float = 1.0):
        '''
        Organises neuron map based on given data.
        :param x_train: Input dataset for training
        :param y_train: Output dataset for training
        :param epochs: Number of epochs
        :param learning_rate: Learning rate used for updating weights
        :param std: Standard deviation for gaussian neighborhood function
        :return:
        '''
        self.w_map = np.random.rand(self.map_shape[0], self.map_shape[1], x_train.shape[1])
        lr = learning_rate
        s = std

        for epoch in range(epochs):
            xk = x_train[np.random.randint(0, len(x_train))]
            bmu_idx = self.find_bmu(xk)
            for i in range(self.map_shape[0]):
                for j in range(self.map_shape[1]):
                    distance = SOMClassifier.e_distance(np.array(bmu_idx), np.array([i, j]))
                    h = np.exp(-(distance ** 2) / (2 * s ** 2))
                    self.w_map[i, j] += lr * h * (x_train[i, j] - xk)
            lr = learning_rate * (1 - epoch/epochs)
            s = std * (1 - epoch/epochs)

        self.set_labels(x_train, y_train)

    def predict(self, samples):
        '''
        Predicts a label of given data samples.
        :param samples: Data samples
        :return: Data labels
        '''
        results = []
        for sample in samples:
            bmu_idx = self.find_bmu(sample)
            results.append(self.neuron_labels[bmu_idx])
        return results

    def find_bmu(self, xk) -> tuple[int, int]:
        winner = (0, 0)
        shortest_distance = np.inf
        for row in range(self.map_shape[0]):
            for col in range(self.map_shape[1]):
                distance = self.e_distance(self.w_map[row][col], xk)
                if distance < shortest_distance:
                    shortest_distance = distance
                    winner = (row, col)
        return winner

    def set_labels(self, x_train, y_train):
        for row in range(self.map_shape[0]):
            for col in range(self.map_shape[1]):
                distances = [SOMClassifier.e_distance(self.w_map[row, col], x) for x in x_train]
                self.neuron_labels[(row, col)] = y_train[np.argmin(distances)]

    @staticmethod
    def e_distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.sqrt(np.sum((a - b) ** 2))