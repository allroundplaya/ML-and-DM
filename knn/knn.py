import numpy as np
import math
from collections import Counter
from scipy.spatial import distance as dist

class KNN():
    def __init__(self, k, dist_type):
        """
        :param k:
        :param dist_type:
            Data Type: Character
            "E" or "e": Euclidean distance
            "M" or "m": Manhattan distance
            "L" or "l": L∞ distance
        """
        self.k = k
        self.dist_type = dist_type
        self.train_data = np.array([])

    def train(self, train_data):
        self.train_data = train_data

    def get_distance(self, test_data):
        """
        Calculates and returns distance between trained data and test data.
        :param test_data:
            Data Type: 1d numpy array
            new data

        :return:
            Data Type: 1d numpy array
            distance between trained data and test data
        """
        distance = np.array([])

        # Euclidean
        if self.dist_type == "E" or self.dist_type == 'e':
            temp = self.train_data[:, 1:] - test_data[1:]
            temp **= 2
            for line in temp:
                distance = np.append(distance, math.sqrt((line.sum())))

        # Manhattan
        if self.dist_type == "M" or self.dist_type == 'm':
            temp = abs(self.train_data[:, 1:] - test_data[1:])
            for i in range(len(temp)):
                distance = np.append(distance, np.sum(temp[i]))

        # L infinity
        if self.dist_type == "L" or self.dist_type == 'l':
            # temp = abs(self.train_data[:, 1:] - test_data[1:])
            # for i in range(len(temp)):
            #     distance = np.append(distance, np.max(temp[i]))
            for row in self.train_data[:, 1:]:
                distance = np.append(distance, dist.chebyshev(row, test_data[1:]))

        return distance

    def predict(self, test_data):
        """
        if you want to use this func for all test data, use enhanced for loop and append func.
        :param test_data:
            Data Type: 1d numpy array
            the record that is used to test the classifier
        :return:
        """
        # print(test_data)
        # print(self.get_distance(test_data))
        closest_idx = np.argsort(self.get_distance(test_data))[:self.k]
        # print(closest_idx)
        k_label = np.array([])
        for idx in closest_idx:
            k_label = np.append(k_label, self.train_data[idx][0])

        cnt = Counter(k_label)
        # print(k_label)
        # print(cnt.most_common())
        return cnt.most_common()[0][0]

