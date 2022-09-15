import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal

from preprocessing.missing_values import MissingValuesInserter


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


def test_raising_when_passed_python_list():
    with pytest.raises(ValueError):
        MissingValuesInserter(columns=[0]).fit_transform([1, 2, 3])


def test_raising_when_passed_no_columns(data):
    X, y = data
    with pytest.raises(TypeError):
        MissingValuesInserter().fit_transform(X)


def test_percentage_numpy(data):
    X, y = data
    no_objects = X.shape[0]
    print(X.shape)
    columns = [0, 3]
    percentage = 0.25
    expected_no_nans = np.zeros(shape=(X.shape[1]))
    print(expected_no_nans)
    for i in columns:
        expected_no_nans.put(i, round(no_objects * 0.25))
    print(expected_no_nans)
    transformer = MissingValuesInserter(columns=columns, percentage=percentage)
    missing = transformer.fit_transform(X)
    print(~np.isnan(missing))
    no_nans = np.count_nonzero(np.isnan(missing), axis=0)
    print(no_nans)
    assert_array_equal(no_nans, expected_no_nans)


def test_percentage_dataframe(data):
    X,y = data
    df = pd.DataFrame(data=X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    print(df)
    no_objects = X.shape[0]
    print(X.shape)
    columns = ['sepal_length', 'petal_length']
    columns_indexes = [0, 2]
    percentage = 0.25
    expected_no_nans = np.zeros(shape=(X.shape[1]))
    print(expected_no_nans)
    for i in columns_indexes:
        expected_no_nans.put(i, round(no_objects * 0.25))
    print(expected_no_nans)
    transformer = MissingValuesInserter(columns=columns, percentage=percentage)
    missing = transformer.fit_transform(df)
    print(~np.isnan(missing))
    no_nans = np.count_nonzero(np.isnan(missing), axis=0)
    print(no_nans)
    assert_array_equal(no_nans, expected_no_nans)


def test_one_column(data):
    X, y = data
    df = pd.DataFrame(data=X, columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    no_objects = X.shape[0]
    columns = ('sepal_length')
    columns_indexes = [0]
    percentage = 0.25
    expected_no_nans = np.zeros(shape=(X.shape[1]))
    print(expected_no_nans)
    for i in columns_indexes:
        expected_no_nans.put(i, round(no_objects * 0.25))
    print(expected_no_nans)
    transformer = MissingValuesInserter(columns=columns, percentage=percentage)
    missing = transformer.fit_transform(df)
    print(missing)
    no_nans = np.count_nonzero(np.isnan(missing), axis=0)
    print(no_nans)
    assert_array_equal(no_nans, expected_no_nans)
