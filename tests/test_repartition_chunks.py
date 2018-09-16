import concurrent.futures
import logging
import pytest
import zap.base as np  # zap includes everything in numpy, with some overrides and new functions
import zap.executor.array
import zap.direct.array
import zap.spark.array

from pyspark.sql import SparkSession

TESTS = [0, 1, 2]


class TestZapArray:
    @pytest.fixture()
    def x(self):
        return np.array([[a] for a in range(12)])

    @pytest.fixture(scope="module")
    def sc(self):
        logger = logging.getLogger("py4j")
        logger.setLevel(logging.WARN)
        spark = (
            SparkSession.builder.master("local[2]")
            .appName("my-local-testing-pyspark-context")
            .getOrCreate()
        )
        yield spark.sparkContext
        spark.stop()

    @pytest.fixture(params=TESTS)
    def xd(self, sc, x, request):
        if request.param == 0:
            yield zap.direct.array.from_ndarray(x.copy(), (5, 1))
        elif request.param == 1:
            yield zap.spark.array.from_ndarray(sc, x.copy(), (5, 1))
        elif request.param == 2:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.from_ndarray(executor, x.copy(), (5, 1))

    @pytest.fixture(params=TESTS)
    def xd34(self, sc, x, request):
        if request.param == 0:
            yield zap.direct.array.from_ndarray(x.copy(), (3, 4))
        elif request.param == 1:
            yield zap.spark.array.from_ndarray(sc, x.copy(), (3, 4))
        elif request.param == 2:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.from_ndarray(executor, x.copy(), (3, 4))

    @pytest.fixture(params=TESTS)
    def xd43(self, sc, x, request):
        if request.param == 0:
            yield zap.direct.array.from_ndarray(x.copy(), (4, 3))
        elif request.param == 1:
            yield zap.spark.array.from_ndarray(sc, x.copy(), (4, 3))
        elif request.param == 2:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.from_ndarray(executor, x.copy(), (4, 3))

    def check(self, expected_rows, actual_rows):
        assert len(actual_rows) == len(expected_rows)
        for i in range(len(expected_rows)):
            from numpy.testing import assert_array_equal

            assert_array_equal(expected_rows[i], actual_rows[i])

    def test_no_op(self, x, xd):
        xd = xd._repartition_chunks((5, 1))
        assert xd.partition_row_counts == [5, 5, 2]
        expected_rows = [
            np.array([[0], [1], [2], [3], [4]]),
            np.array([[5], [6], [7], [8], [9]]),
            np.array([[10], [11]]),
        ]
        actual_rows = xd._compute()
        self.check(expected_rows, actual_rows)

    def test_uneven(self, x, xd):
        subset = np.array([True] * 12)
        subset[7] = False  # drop a row
        xd = xd[subset, :]
        xd = xd._repartition_chunks((5, 1))
        assert xd.partition_row_counts == [5, 5, 1]
        expected_rows = [
            np.array([[0], [1], [2], [3], [4]]),
            np.array([[5], [6], [8], [9], [10]]),
            np.array([[11]]),
        ]
        actual_rows = xd._compute()
        self.check(expected_rows, actual_rows)

    def test_subdivide(self, x, xd34):
        xd = xd34._repartition_chunks((2, 1))
        assert xd.partition_row_counts == [2, 2, 2, 2, 2, 2]
        expected_rows = [
            np.array([[0], [1]]),
            np.array([[2], [3]]),
            np.array([[4], [5]]),
            np.array([[6], [7]]),
            np.array([[8], [9]]),
            np.array([[10], [11]]),
        ]
        actual_rows = xd._compute()
        self.check(expected_rows, actual_rows)

    def test_coalesce(self, x, xd43):
        xd = xd43._repartition_chunks((6, 1))
        assert xd.partition_row_counts == [6, 6]
        expected_rows = [
            np.array([[0], [1], [2], [3], [4], [5]]),
            np.array([[6], [7], [8], [9], [10], [11]]),
        ]
        actual_rows = xd._compute()
        self.check(expected_rows, actual_rows)
