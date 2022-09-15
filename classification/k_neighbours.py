import random

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from statistics import mean
import pandas as pd
import numpy as np
from dataset.aggregations import A1Aggregation


class KNNAlgorithmM(BaseEstimator, ClassifierMixin):
    def __init__(self, k_neighbours=(3, 5, 7), metric='euclidean', main_class=1, missing_representation=-1):
        self.k_neighbours = k_neighbours
        self.metric = metric
        self.main_class = main_class
        self.missing_representation = missing_representation

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.p_ = []
        return self

    def difference_with_missing_values(self, x, y):
        if x == self.missing_representation and y == self.missing_representation:
            return np.array([1])
        if x == self.missing_representation:
            return np.array([np.maximum(y, 1 - y)])
        if y == self.missing_representation:
            return np.maximum(x, 1 - x)
        return np.array([np.abs(x - y)])

    def euclidean_distance_with_missing_values(self, test, train):
        diff = np.vectorize(self.difference_with_missing_values, otypes=[np.float64])(test, train)
        return np.sqrt(np.sum(np.square(diff, dtype=np.float64), axis=0, dtype=np.float64), dtype=np.float64)

    def euclidean_distance_with_missing_values_optimized(self, test, train):
        if len(test.shape) != 1:
            diff = np.vectorize(self.difference_with_missing_values, otypes=[np.float64])(test[:, np.newaxis], train)
            return np.sqrt(np.sum(np.square(diff, dtype=np.float64), axis=2, dtype=np.float64), dtype=np.float64)
        if len(test.shape) == 1:
            diff = np.vectorize(self.difference_with_missing_values, otypes=[np.float64])(test, train)
            return np.sqrt(np.sum(np.square(diff, dtype=np.float64), axis=0, dtype=np.float64), dtype=np.float64)

    def get_missing_values_indexes(self, X):
        missing_indexes = {}
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                if X[i, j] == self.missing_representation:
                    if not str(i) in missing_indexes.keys():
                        missing_indexes[str(i)] = [j]
                    else:
                        missing_indexes[str(i)].append(j)
        return missing_indexes

    def sort_distances(self, distance):
        """
        :param distance: a numpy 2D array, holding a distances of test objects from training objects
        :return: a numpy 2D array, holding indexes of objects
        this method sorts distances of test objects from train objects and then
        change distance to index of object
        """
        return np.argsort(distance, axis=1)

    def take_k_smallest_and_sort_distances(self, distance, k):
        """
        :param distance:  a numpy 2D array, holding a distances of test objects from training objects
        :param k: number of nearest objects to find
        :return:
        This method first finds k-smallest distances between test object and train objects, then
        sorts this k-distances only. Finally the distances are changed into indexes of train objects.
        """
        no_rows = distance.shape[0]
        partitioned = np.argpartition(distance, axis=1, kth=range(k))[:, :k]
        return partitioned

    def _classic_knn_optimized(self, sorted_distance, k_neighbours):
        if k_neighbours > sorted_distance[0].size:
            raise ValueError('k =', k_neighbours, ' is bigger than a size of data !!! (data)= ', sorted_distance[0].size)
        nearest_neighbours = sorted_distance[:, 0:k_neighbours]
        record = self.y_[nearest_neighbours]
        return np.sum(record == 1, axis=1) / k_neighbours

    def _classic_knn(self, sorted_distance, k_neighbours, test_object_index):
        if k_neighbours > sorted_distance[0].size:
            raise ValueError('k =', k_neighbours, ' is bigger than a size of data !!! (data)= ', sorted_distance[0].size)
        # take k-nearest neighbours
        nearest_neighbours = sorted_distance[test_object_index, 0:k_neighbours]
        record = self.y_[nearest_neighbours]
        # create a temporary data frame to count decisions classes of k-neighbours
        dataFrame = pd.DataFrame({'neighbour_decision': record})
        decision_class_count = dataFrame['neighbour_decision'].value_counts().to_frame()
        if 1 not in decision_class_count.index.values:
            main_class_probability = 0
        else:
            main_class_probability = decision_class_count.loc[1, 'neighbour_decision'] / k_neighbours
        return main_class_probability

    def compute_final_p(self, X):
        distance = self.euclidean_distance_with_missing_values_optimized(X, self.X_)
        sorted = self.take_k_smallest_and_sort_distances(distance, max(self.k_neighbours))
        self.p_ = np.empty(shape=(X.shape[0], len(self.k_neighbours)))
        i = 0
        for k in self.k_neighbours:
            self.p_[:, i] = self._classic_knn_optimized(sorted, k)
            i += 1
        return self.p_

    def predict_optimized(self, X):
        """
        This method is optimized version of predict method
        :param X:
        :return:
        """
        final_p = self.compute_final_p(X)
        mean_final_p = np.mean(final_p, axis=1)
        predicted_decision = np.zeros((X.shape[0]), dtype=int)
        positive_decision_indexes = np.argwhere(mean_final_p > 0.5)
        if positive_decision_indexes.shape[0] != 0:
            predicted_decision[positive_decision_indexes] = 1
        return predicted_decision

    def predict(self, X):
        """
        This method is unoptimized version, directly following algorithm pseudocode
        :param X:
        :return:
        """
        check_is_fitted(self)
        X = check_array(X)
        #return self.predict_optimized(X)

        distance = np.empty(shape=(X.shape[0], self.X_.shape[0]), dtype=np.float64)
        i = j = 0
        for test in X:
            for train in self.X_:
                distance[i][j] = self.euclidean_distance_with_missing_values(test, train)
                j += 1
            i += 1
            j = 0
        sorted = np.argsort(distance, axis=1)
        print(sorted)
        self.p_ = np.zeros(shape=(X.shape[0], len(self.k_neighbours)))
        j = 0
        for i in range(0, X.shape[0]):
            j = 0
            for k in self.k_neighbours:
                self.p_[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        predicted_decision = np.ndarray(shape=(X.shape[0],))
        i = 0
        for record_p in self.p_:
            if mean(record_p) > 0.5:
                predicted_decision[[i]] = 1
            else:
                predicted_decision[[i]] = 0
            i += 1

        return predicted_decision

    def predict_proba_optimized(self, X):
        final_p = self.compute_final_p(X)
        predicted_decision_proba = np.empty(shape=(X.shape[0], 2), dtype=float)
        mean_final_p = np.mean(final_p, axis=1)
        predicted_decision_proba[:, 1] = mean_final_p
        predicted_decision_proba[:, 0] = 1 - mean_final_p
        return predicted_decision_proba

    def predict_proba(self, X):
        predicted_decision_proba = np.ndarray(shape=(X.shape[0], 2))
        check_is_fitted(self)
        X = check_array(X)
        distance = np.empty(shape=(X.shape[0], self.X_.shape[0]), dtype=np.float64)
        i = j = 0
        for test in X:
            for train in self.X_:
                distance[i][j] = self.euclidean_distance_with_missing_values(test, train)
                j += 1
            i += 1
            j = 0
        sorted = np.argsort(distance, axis=1)
        self.p_ = np.zeros(shape=(X.shape[0], len(self.k_neighbours)))
        j = 0
        for i in range(0, X.shape[0]):
            j = 0
            for k in self.k_neighbours:
                self.p_[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        i = 0
        for record_p in self.p_:
            predicted_decision_proba[i, 1] = mean(record_p)
            predicted_decision_proba[i, 0] = 1 - predicted_decision_proba[i, 1]
            i += 1

        return predicted_decision_proba


class KNNAlgorithmF(KNNAlgorithmM):
    def __init__(self, k_neighbours=(3, 5, 7), r=10, aggregation=A1Aggregation(), missing_representation=-1):
        self.k_neighbours = k_neighbours
        self.r = r
        self.aggregation = aggregation
        self.missing_representation = missing_representation

    def fit(self, X, y):
        #X, y = check_X_y(X, y)
        self.classes_ = np.array([0, 1]) #unique_labels(y)
        self.X_ = X
        self.y_ = y
        self.p_ = []
        self.attributes_distinct_values = {}
        for i in range(X.shape[1]):
            self.attributes_distinct_values[i] = np.unique(X[:, i])
            self.attributes_distinct_values[i] = np.delete(self.attributes_distinct_values[i],
                                                           np.where(self.attributes_distinct_values[i] == self.missing_representation))
        return self

    def fill_missing_value(self, instance_index, columns_indexes, X):
        instances_filled = np.ndarray(shape=(self.r, self.X_.shape[1]))
        for i in range(self.r):
            instances_filled[i] = X[instance_index].copy()
            for j in columns_indexes:
                if self.attributes_distinct_values[j].shape[0] == 0:
                    instances_filled[i][j] = -1
                else:
                    if self.attributes_distinct_values[j].shape[0] == 1:
                        random_value_from_dataset = 0
                    else:
                        random_value_from_dataset = random.randint(0, self.attributes_distinct_values[j].shape[0] - 1)
                    instances_filled[i][j] = self.attributes_distinct_values[j][random_value_from_dataset]
                #instances_filled[i][j] = self.attributes_distinct_values[j][random_value_from_dataset]
        distance = euclidean_distances(instances_filled, self.X_)
        sorted = np.argsort(distance, axis=1)
        p = np.ndarray(shape=(self.r, len(self.k_neighbours)))
        for i in range(self.r):
            j = 0
            for k in self.k_neighbours:
                p[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        min_p = np.ndarray(shape=(self.r,))
        max_p = np.ndarray(shape=(self.r,))
        fuzzy_sets = np.ndarray(shape=(self.r, 2))
        for i in range(self.r):
            min_p[i] = np.min(p[i])
            max_p[i] = np.max(p[i])
            fuzzy_sets[i] = np.array([min_p[i], max_p[i]])

        uncertainty_interval = self.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets)
        final_p = uncertainty_interval.sum() / 2
        return final_p

    def fill_missing_value_and_classify_optimized(self, X):
        missing_values_indexes = np.argwhere(X == self.missing_representation)
        instances_filled = np.ndarray(shape=(missing_values_indexes.shape[0], self.r, self.X_.shape[1]))
        instances_filled[missing_values_indexes[:, 0], :] = np.repeat(X[missing_values_indexes[:, 0], np.newaxis], self.r, axis=1)
        instances_filled[:, 0, missing_values_indexes[:, 1]] = [d[random.randint(0, d.shape[0] - 1)] for d in
                                          self.attributes_distinct_values[missing_values_indexes[:, 1]]]
        self.compute_final_p(instances_filled[:])

    def fill_missing_value_and_classify(self, instance_index, columns_indexes, X):
        instances_filled = np.ndarray(shape=(self.r, self.X_.shape[1]))
        #print(instance_index)
        for i in range(self.r):
            instances_filled[i] = X[instance_index].copy()
            for j in columns_indexes:
                random_value_from_dataset = random.randint(0, self.attributes_distinct_values[j].shape[0]-1)
                instances_filled[i][j] = self.attributes_distinct_values[j][random_value_from_dataset]
        distance = euclidean_distances(instances_filled, self.X_)
        sorted = np.argsort(distance, axis=1)
        p = np.ndarray(shape=(self.r, len(self.k_neighbours)))
        for i in range(self.r):
            j = 0
            for k in self.k_neighbours:
                p[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        min_p = np.ndarray(shape=(self.r,))
        max_p = np.ndarray(shape=(self.r,))
        fuzzy_sets = np.ndarray(shape=(self.r, 2))
        for i in range(self.r):
            min_p[i] = np.min(p[i])
            max_p[i] = np.max(p[i])
            fuzzy_sets[i] = np.array([min_p[i], max_p[i]])

        uncertainty_interval = self.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets)
        final_p = uncertainty_interval.sum() / 2
        if final_p > 0.5:
            return 1
        else:
            return 0

    def compute_final_p_opt(self, X):
        distance = euclidean_distances(X, self.X_)
        sorted = self.take_k_smallest_and_sort_distances(distance, k=max(self.k_neighbours))

        self.p_ = np.empty(shape=(X.shape[0], len(self.k_neighbours)))
        i = 0
        for k in self.k_neighbours:
            self.p_[:, i] = self._classic_knn_optimized(sorted, k)
            i += 1

        def temp(x, i):
            return np.array([x[:, i], x[:, i]])

        fuzzy_sets = np.empty((X.shape[0], len(self.k_neighbours), 2))
        for i in range(len(self.k_neighbours)):
            fuzzy_sets[:, i] = temp(self.p_, i).T

        uncertainty_interval = np.empty((X.shape[0], 2))
        #uncertainty_interval = np.apply_along_axis(self.aggregation.aggregate, 1, fuzzy_sets)
        uncertainty_interval = fuzzy_sets.sum(axis=1) / fuzzy_sets.shape[1]
        final_p = uncertainty_interval.sum(axis=1) / 2
        return final_p

    def compute_final_p(self, X):
        distance = euclidean_distances(X, self.X_)
        sorted = self.take_k_smallest_and_sort_distances(distance, k=max(self.k_neighbours))
        self.p_ = np.empty(shape=(X.shape[0], len(self.k_neighbours)))
        i = 0
        for k in self.k_neighbours:
            self.p_[:, i] = self._classic_knn_optimized(sorted, k)
            i += 1

        def temp(x, i):
            return np.array([x[:, i], x[:, i]])

        fuzzy_sets = np.empty((X.shape[0], len(self.k_neighbours), 2))
        for i in range(len(self.k_neighbours)):
            fuzzy_sets[:, i] = temp(self.p_, i).T

        uncertainty_interval = np.empty((X.shape[0], 2))
        for i in range(X.shape[0]):
            uncertainty_interval[i] = self.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets[i])
        final_p = uncertainty_interval.sum(axis=1) / 2
        return final_p

    def predict_optimized(self, X):
        records_with_missing_values_indexes = self.get_missing_values_indexes(X)
        predicted_decision = np.zeros(shape=(X.shape[0],), dtype=int)
        final_p = self.compute_final_p(X)
        predicted_decision[np.argwhere(final_p > 0.5)] = 1
        for index in records_with_missing_values_indexes.keys():
            predicted_decision[int(index)] = self.fill_missing_value_and_classify(int(index),
                                                                                  records_with_missing_values_indexes[
                                                                                      index], X)
        return predicted_decision

    def predict_proba_optimized(self, X):
        records_with_missing_values_indexes = self.get_missing_values_indexes(X)
        predicted_decision_proba = np.empty(shape=(X.shape[0], 2), dtype=float)
        final_p = self.compute_final_p(X)
        predicted_decision_proba[:, 1] = final_p
        predicted_decision_proba[:, 0] = 1 - final_p
        for index in records_with_missing_values_indexes.keys():
            temp = self.fill_missing_value(int(index), records_with_missing_values_indexes[index], X)
            predicted_decision_proba[int(index), 1] = temp
            predicted_decision_proba[int(index), 0] = 1 - temp
        return predicted_decision_proba

    def predict(self, X):
        return self.predict_optimized(X)

    def predict_not_optimized(self, X):
        records_with_missing_values_indexes = self.get_missing_values_indexes(X)
        distance = euclidean_distances(X, self.X_)
        sorted = np.argsort(distance, axis=1)
        self.p_ = np.zeros(shape=(X.shape[0], len(self.k_neighbours)))

        for i in range(X.shape[0]):
            j = 0
            for k in self.k_neighbours:
                self.p_[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        predicted_decision = np.ndarray(shape=(X.shape[0],))
        fuzzy_sets = np.zeros((len(self.k_neighbours), 2))
        for i in range(X.shape[0]):
            k = 0
            for j in range(len(self.k_neighbours)):
                fuzzy_sets[k] = np.array([self.p_[i, j], self.p_[i, j]])
                k += 1
            uncertainty_interval = self.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets)
            final_p = uncertainty_interval.sum() / 2
            if final_p > 0.7:
                predicted_decision[i] = 1
            else:
                predicted_decision[i] = 0
        for index in records_with_missing_values_indexes.keys():
            predicted_decision[int(index)] = self.fill_missing_value_and_classify(int(index),
                                                                                  records_with_missing_values_indexes[index], X)
        return predicted_decision

    def predict_proba(self, X):
        return self.predict_proba_optimized(X)

    def predict_proba_not_optimized(self, X):
        predicted_decision_proba = np.ndarray(shape=(X.shape[0], 2))
        records_with_missing_values_indexes = self.get_missing_values_indexes(X)
        distance = euclidean_distances(X, self.X_)
        sorted = np.argsort(distance, axis=1)
        self.p_ = np.zeros(shape=(X.shape[0], len(self.k_neighbours)))

        for i in range(X.shape[0]):
            j = 0
            for k in self.k_neighbours:
                self.p_[i, j] = self._classic_knn(sorted, k, i)
                j += 1

        fuzzy_sets = np.zeros((len(self.k_neighbours), 2))
        for i in range(X.shape[0]):
            k = 0
            for j in range(len(self.k_neighbours)):
                fuzzy_sets[k] = np.array([self.p_[i, j], self.p_[i, j]])
                k += 1
            uncertainty_interval = self.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets)
            final_p = uncertainty_interval.sum() / 2
            predicted_decision_proba[i, 1] = final_p
            predicted_decision_proba[i, 0] = 1 - final_p

        for index in records_with_missing_values_indexes.keys():
            temp = self.fill_missing_value(int(index), records_with_missing_values_indexes[index], X)
            predicted_decision_proba[int(index), 1] = temp
            predicted_decision_proba[int(index), 0] = 1 - temp
        return predicted_decision_proba
