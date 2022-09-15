import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import euclidean_distances

from classification.k_neighbours import KNNAlgorithmM, KNNAlgorithmF, A1Aggregation
from numpy.testing import assert_array_equal, assert_allclose


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_classifier(data):
    # tests default attributes
    X, y = data
    aggregatedKNN = KNNAlgorithmM()
    assert aggregatedKNN.k_neighbours == (3, 5, 7)

    # tests adding special fields in learning
    aggregatedKNN.fit(X, y)
    assert hasattr(aggregatedKNN, 'classes_')
    assert hasattr(aggregatedKNN, 'X_')
    assert hasattr(aggregatedKNN, 'y_')

    # tests output shape
    y_pred = aggregatedKNN.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_measure():
    # create in-memory dataset
    bmi = pd.DataFrame.from_dict({'height': [1.6, 1.6, 1.62, 1.75, 1.7, 1.8, 1.9],
                                  'weight': [60, 80, 80, 90, 80, 85, 82],
                                  'bmi': [1, 0, 0, 0, 0, 1, 1]})
    print(bmi)
    aggregatedKNN = KNNAlgorithmM(k_neighbours=(1, 3))
    # fit a model
    model = aggregatedKNN.fit(bmi.iloc[:, :2], np.ravel(bmi.iloc[:, 2:3]))
    print(bmi.iloc[:, :2])
    print(np.ravel(bmi.iloc[:, 2:3]))
    # checking first object decision
    pred = model.predict([[1.6, 60]])
    print(pred)
    #assert pred == [1]
    # checking classification score (accuracy here)
    score = model.score([[1.6, 60], [1.59, 61], [1.6, 80]], [1, 1, 0])
    print(score)
    assert score == 1.0


def test_main_class_probability():

    test_decisions = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    expected_p = np.array([[1/3, 2/5, 3/7],
                           [1/3, 3/5, 4/7],
                           [1, 3/5, 5/7]])
    a = KNNAlgorithmM()
    a.y_ = test_decisions
    sorted_distance = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                                [5, 6, 1, 2, 3, 0, 7, 9, 8, 10, 11, 4],
                                [3, 7, 8, 9, 10, 11, 2, 1, 4, 0, 6, 5]])

    p = np.zeros(shape=(3, 3))
    j = 0
    for i in range(0, 3):
        j = 0
        for k in (3, 5, 7):
            p[i, j] = a._classic_knn(sorted_distance, k, i)
            j += 1

    assert_allclose(p, expected_p, rtol=0.1, atol=0.1)


def test_aggregation():
    test_decisions = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1])
    given_p = np.array([[1/3, 2/5, 3/7],
                           [1/3, 3/5, 4/7],
                           [1, 3/5, 5/7]])
    fuzzy_sets = np.zeros((3, 2))
    f = KNNAlgorithmF(aggregation=A1Aggregation())
    uncertainty_intervals = np.zeros(shape=(3,2))
    expected_uncertainty_intervals = np.array([[0.3873, 0.3873], [0.5015, 0.5015], [0.7714, 0.7714]])
    expected_final_ps = np.array([0.3873, 0.5015, 0.7714])
    for i in range(3):
        k = 0
        for j in range(len((3, 5, 7))):
            fuzzy_sets[k] = np.array([given_p[i, j], given_p[i, j]])
            k += 1
        uncertainty_interval = f.aggregation.aggregate_numpy_arrays_representation(fuzzy_sets)
        uncertainty_intervals[i] = uncertainty_interval
        print(uncertainty_interval)
        final_p = uncertainty_interval.sum() / 2
        print(final_p)
    assert_array_equal(uncertainty_intervals, expected_uncertainty_intervals)


def test_computing_euclidean_distance_with_missing_values():
    dummy_train = np.array([[0.2, 0.5, 1.0],
                            [-1, 0.4, 0.1],
                            [0.2, 0.4, 0.4]])

    dummy_test = np.array([[0.8, 0.1, -1],
                           [0.7, 0.2, 0.4],
                           [0.6, -1, 0.8]])

    distance = euclidean_distances(dummy_test, dummy_train)
    print(distance)
    #assert_array_equal(expected_distance, distance)
    missing_indexes = KNNAlgorithmM().get_missing_values_indexes(dummy_test)
    print(missing_indexes)
    missing_train_indexes = KNNAlgorithmM().get_missing_values_indexes(dummy_train)
    print(missing_train_indexes)
    for k in missing_indexes.keys():
        for test in range(dummy_train.shape[0]):
            distance[int(k)][test] = KNNAlgorithmM().euclidean_distance_with_missing_values(dummy_test[test], dummy_train[int(k)])
    print(distance)
    #for k in missing_train_indexes.keys():
    #    for train in range(dummy_test.shape[0]):
    #        distance[train][int(k)] = KNNAlgorithmM.euclidean_distance_with_missing_values(dummy_train[int(k)], dummy_test[train])
    #print(distance)

    i = j = 0
    for test in dummy_test:
        for train in dummy_train:
            distance[i][j] = KNNAlgorithmM().euclidean_distance_with_missing_values(test, train)
            j += 1
        i += 1
        j = 0
    print(distance)

    expected_distance = np.array([[1.232, 1.240, 0.9],
                                  [0.836, 0.787, 0.538],
                                  [0.67, 1.191, 0.824]])
    assert_array_equal(expected_distance, distance)


def test_sorting():
    distance = np.array([[0.8653, 0.6731, 0.9863, 0.3042, 0.9, 0.452],
                         [0.1231, 0.8763, 0.0112, 0.7633, 0.912, 0.341]])
    tested = KNNAlgorithmM().take_k_smallest_and_sort_distances(distance, 5)

    assert_array_equal(tested, np.argsort(distance)[:, :5])


def test_compute_final_p():
    X, y = load_iris(True)
    f = KNNAlgorithmF()
    f.fit(X, y)
    assert_array_equal(f.compute_final_p(X), f.compute_final_p_opt(X))


def test_m_dist_equivalence():
    X, y = load_iris(True)
    m = KNNAlgorithmM()
    m.fit(X, y)

    def full_distance(X, X_):
        distance = np.empty(shape=(X.shape[0], X_.shape[0]), dtype=np.float64)
        i = j = 0
        for test in X:
            for train in X_:
                distance[i][j] = m.euclidean_distance_with_missing_values(test, train)
                j += 1
            i += 1
            j = 0
        return distance

    #assert_array_equal(m.euclidean_distance_with_missing_values(X, X),
    #                   m.euclidean_distance_with_missing_values_optimized(X, X))
    assert_array_equal(m.euclidean_distance_with_missing_values_optimized(X, X),
                       full_distance(X, X))
