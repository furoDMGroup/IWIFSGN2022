import pytest

from sklearn.utils.estimator_checks import check_estimator
from classification.k_neighbours import KNNAlgorithmM


@pytest.mark.parametrize(
    "Estimator", [KNNAlgorithmM()]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)