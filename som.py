import random
import numpy as np
from collections import defaultdict, Counter


class SOMClassifier:
    def __init__(self, map_shape: tuple[int, int] = (10, 10)):
        self.map_shape = map_shape
        self.w_map = np.array([])
        self.neuron_labels = np.zeros((10, 10))

    def learn(self, training_x: np.array, training_y: np.array,
              learning_rate: float = 0.01, iterations: int = 1000):
        """
        Organises neuron map based on given data.
        :param training_x: Data for training
        :param training_y: Data labels
        :param learning_rate: Number of learning rate
        :param iterations: Number of iterations
        """
        # initializing map
        w_map = []
        for i in range(self.map_shape[0]):
            w_row = []
            for j in range(self.map_shape[1]):
                w_row.append(np.zeros(training_x.shape[1]))
            w_map.append(np.array(w_row))
        w_map = np.array(w_map)
        best_neuron_idx = (0, 0)

        # training: finding best matching unit and updating surrounding neurons
        while np.any(np.sign(w_map[best_neuron_idx].dot(training_x.transpose())) != training_y)\
                and iterations > 0:
            xk = random.choice(training_x)
            best_neuron_idx = self.find_best_neuron(w_map, xk)
            print(best_neuron_idx)

            h = SOMClassifier.h_function(self.map_shape, best_neuron_idx, 2)
            for i in range(self.map_shape[0]):
                for j in range(self.map_shape[1]):
                    w_map[i][j] = w_map[i][j] + learning_rate * h[i][j] * (xk - w_map[i][j])
            iterations -= 1
            learning_rate -= 0.000001
        self.w_map = w_map
        print(self.w_map)

        # creating label matrix
        labels = defaultdict(list)
        for i in range(training_x.shape[0]):
            best_neuron_idx = self.find_best_neuron(w_map, training_x[i])
            labels[best_neuron_idx].append(training_y[i])

        for i, labels in labels.items():
            self.neuron_labels[i] = Counter(labels).most_common(1)[0][0]

    def predict(self, samples: np.array):
        """
        Predicts a label of given data samples.
        :param samples: Data samples
        :return: Data labels
        """
        y = []
        for sample in samples:
            best_neuron_idx = self.find_best_neuron(self.w_map, sample)
            y.append(int(self.neuron_labels[best_neuron_idx]))
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

    @staticmethod
    def h_function(grid_shape: tuple[int, int],
                   winner_coords,
                   stand_deviation: float) -> np.ndarray:
        """
        Neighborhood function. Determines how much neurons are updated
        based on a distance from best matching unit.
        :param grid_shape: Neuron map shape
        :param winner_coords: (x, y) coordinates
        :param stand_deviation: Standard deviation
        :return:
        """
        h_indices = np.indices(grid_shape).transpose(1, 2, 0)
        e_distances = np.sqrt(np.sum((h_indices - winner_coords)**2, axis=-1))
        return np.exp(-e_distances / (2*(stand_deviation**2)) )