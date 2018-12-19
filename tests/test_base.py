import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from zappy.base import ZappyArray

testdata = [
    (
        np.array([0, 1, 2]),
        np.array([2, 1]),
        np.array([np.array([0, 1]), np.array([0])]),
    ),
    (np.array([0, 1]), np.array([2, 1]), np.array([np.array([0, 1]), np.array([])])),
    (np.array([1, 2]), np.array([2, 1]), np.array([np.array([1]), np.array([0])])),
    (
        np.array([1, 5]),
        np.array([2, 3, 2, 1]),
        np.array([np.array([1]), np.array([]), np.array([0]), np.array([])]),
    ),
    (
        slice(None),
        np.array([2, 3, 2, 1]),
        np.array([slice(0, None), slice(None), slice(None), slice(1)]),
    ),
    (
        slice(2, 5),
        np.array([2, 3, 2, 1]),
        np.array([slice(0, 0), slice(0, 3), slice(0, 0), slice(0, 0)]),
    ),
    (
        slice(1, 6),
        np.array([2, 3, 2, 1]),
        np.array([slice(1, None), slice(None), slice(1), slice(0, 0)]),
    ),
]


@pytest.mark.parametrize("subset,partition_row_counts,expected", testdata)
def test_copartition(subset, partition_row_counts, expected):
    actual = ZappyArray._copartition(subset, partition_row_counts)
    assert len(actual) == len(expected)
    for i, arr in enumerate(actual):
        assert_array_equal(arr, expected[i])
