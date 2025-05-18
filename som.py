import numpy as np


class SOMClassifier:
    """
    Implementation of Self-Organizing Map
    """

    def __init__(self, map_shape: tuple[int, int] = (7, 7)):
        self.map_shape = map_shape
        self.w_map = np.array([])
        self.neuron_labels = {}

    def learn(self, training_x: np.array, training_y: np.array, epochs: int,
              learning_rate: float = 0.5, standard_deviation: float = 2):
        """
        Organises neuron map based on given data.
        :param training_x: Data for training
        :param training_y: Data labels
        :param learning_rate: Learning rate
        :param standard_deviation: Standard deviation
        :param epochs: Number of epochs
        """
        lr0 = learning_rate
        s0 = standard_deviation
        quantisation_error = np.empty(int(epochs / 10))

        # initializing random map
        w_map = np.random.rand(self.map_shape[0], self.map_shape[1], training_x.shape[1])

        # training: finding best matching unit and updating surrounding neurons
        for epoch in range(epochs):
            xk = training_x[np.random.randint(0, len(training_x))]
            best_neuron_idx = self.find_best_neuron(w_map, xk)

            # # updating map
            w_map = self.update_map(w_map, standard_deviation, learning_rate, best_neuron_idx, xk)
            learning_rate = lr0 * (1 - epoch / epochs)
            standard_deviation = s0 * (1 - epoch / epochs)

            # calculating error
            if epoch % 10 == 0:
                bmu = [w_map[self.find_best_neuron(w_map, x)] for x in training_x]
                errors = [np.linalg.norm(x - bmu_vec) for x, bmu_vec in zip(training_x, bmu)]
                quantisation_error[int(epoch / 10)] = np.mean(errors)
        self.w_map = w_map

        # creating label matrix
        self.create_labels(training_x, training_y, w_map)
        return quantisation_error

    def update_map(self, w_map: np.array, sigma: float,
                   lr: float, idx, x) -> np.array:
        """
        Updates map using neighborhood function and learning rate.
        :param w_map: Neuron map
        :param sigma: Standard deviation
        :param lr: Learning rate
        :param idx: Best neuron index
        :param x: Data sample
        :return: Updated neuron map
        """
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                d = np.sqrt((i - idx[0]) ** 2 + (j - idx[1]) ** 2)
                h = np.exp(-(d ** 2) / (2 * sigma ** 2))
                w_map[i, j] += lr * h * (x - w_map[i, j])
        return w_map

    def predict(self, samples: np.array):
        """
        Predicts a label of given data samples.
        :param samples: Data samples
        :return: Data labels
        """
        y = []
        for sample in samples:
            best_neuron_idx = self.find_best_neuron(self.w_map, sample)
            if best_neuron_idx in self.neuron_labels:
                y.append(int(self.neuron_labels[best_neuron_idx]))
            else:
                min_dist = float('inf')
                for neuron, label in self.neuron_labels.items():
                    dist = SOMClassifier.e_distance(neuron, best_neuron_idx)
                    if dist < min_dist:
                        min_dist = dist
                        best_label = label
                y.append(best_label)
        return np.array(y)

    def find_best_neuron(self, w_map: np.ndarray, xk):
        """
        Finds best matching unit's coordinates based on Euclidean distance
        between a neuron and a data sample.
        :param w_map: Neuron map
        :param xk: Data sample
        :return: (x, y) coordinates
        """
        e_distances = []
        for i in range(self.map_shape[0]):
            temp = []
            for j in range(self.map_shape[1]):
                temp.append(SOMClassifier.e_distance(w_map[i][j], xk))
            e_distances.append(temp)
        e_distances = np.array(e_distances)
        return np.unravel_index(np.argmin(e_distances), e_distances.shape)

    @staticmethod
    def e_distance(w: np.array, xi) -> float:
        """
        Calculates Euclidean distance between two vectors.
        :param w: First vector
        :param xi: Second vector
        :return: Euclidean distance
        """
        result = 0
        for i in range(len(w)):
            result += np.power(w[i] - xi[i], 2)
        return np.sqrt(result)

    def create_labels(self, training_x: np.array,
                      training_y: np.array, w_map: np.array):
        """
        Creates labels of trained map based on training data and it's labels.
        Params training_x and training_y must have same size for axis=1.
        :param training_x: Normalised training data
        :param training_y: Labels of training data
        :param w_map: Trained neuron map
        """
        neuron_labels = {}

        for i, (x, label) in enumerate(zip(training_x, training_y)):
            best_neuron_idx = self.find_best_neuron(w_map, x)
            if best_neuron_idx not in neuron_labels:
                neuron_labels[best_neuron_idx] = []
            neuron_labels[best_neuron_idx].append(label)

        for i, labels in neuron_labels.items():
            neuron_labels[i] = max(set(labels), key=labels.count)

        self.neuron_labels = neuron_labels
