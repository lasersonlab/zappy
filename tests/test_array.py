import concurrent.futures
import logging
import pytest
import sys
import zap.base as np  # zap includes everything in numpy, with some overrides and new functions
import zap.executor.array
import zap.direct.array
import zap.spark.array
import zarr

from numpy.testing import assert_allclose
from pyspark.sql import SparkSession

TESTS = [0, 1, 2, 3, 6, 7]
ZEROS_TESTS = [0, 1, 2]
ONES_TESTS = ZEROS_TESTS

# only run Beam tests on Python 2, and don't run executor tests
if sys.version_info[0] == 2:
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    import zap.beam.array

    TESTS = [0, 1, 2, 3, 4, 5]


class TestZapArray:
    @pytest.fixture()
    def x(self):
        return np.array(
            [
                [0.0, 1.0, 0.0, 3.0, 0.0],
                [2.0, 0.0, 3.0, 4.0, 5.0],
                [4.0, 0.0, 0.0, 6.0, 7.0],
            ]
        )

    @pytest.fixture()
    def chunks(self):
        return (2, 5)

    @pytest.fixture()
    def xz(self, x, chunks, tmpdir):
        input_file_zarr = str(tmpdir.join("x.zarr"))
        z = zarr.open(
            input_file_zarr, mode="w", shape=x.shape, dtype=x.dtype, chunks=chunks
        )
        z[:] = x.copy()  # write as zarr locally
        return input_file_zarr

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
    def xd(self, sc, x, xz, chunks, request):
        if request.param == 0:
            # zarr direct
            yield zap.direct.array.from_zarr(xz)
        elif request.param == 1:
            # in-memory ndarray direct
            yield zap.direct.array.from_ndarray(x.copy(), chunks)
        elif request.param == 2:
            # zarr spark
            yield zap.spark.array.from_zarr(sc, xz)
        elif request.param == 3:
            # in-memory ndarray spark
            yield zap.spark.array.from_ndarray(sc, x.copy(), chunks)
        elif request.param == 4:
            # zarr beam
            pipeline_options = PipelineOptions()
            pipeline = beam.Pipeline(options=pipeline_options)
            yield zap.beam.array.from_zarr(pipeline, xz)
        elif request.param == 5:
            # in-memory ndarray beam
            pipeline_options = PipelineOptions()
            pipeline = beam.Pipeline(options=pipeline_options)
            yield zap.beam.array.from_ndarray(pipeline, x.copy(), chunks)
        elif request.param == 6:
            # zarr executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.from_zarr(executor, xz)
        elif request.param == 7:
            # in-memory ndarray executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.from_ndarray(executor, x.copy(), chunks)

    @pytest.fixture(params=ZEROS_TESTS)
    def zeros(self, sc, request):
        if request.param == 0:
            yield zap.direct.array.zeros((3, 5), chunks=(2, 5), dtype=int)
        elif request.param == 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.zeros(
                    executor, (3, 5), chunks=(2, 5), dtype=int
                )
        elif request.param == 2:
            yield zap.spark.array.zeros(sc, (3, 5), chunks=(2, 5), dtype=int)
        # TODO beam

    @pytest.fixture(params=ONES_TESTS)
    def ones(self, sc, request):
        if request.param == 0:
            yield zap.direct.array.ones((3, 5), chunks=(2, 5), dtype=int)
        elif request.param == 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.ones(
                    executor, (3, 5), chunks=(2, 5), dtype=int
                )
        elif request.param == 2:
            yield zap.spark.array.ones(sc, (3, 5), chunks=(2, 5), dtype=int)
        # TODO beam

    def test_identity(self, x, xd):
        assert_allclose(xd.asndarray(), x)

    def test_scalar_arithmetic(self, x, xd):
        xd = (((xd + 1) * 2) - 4) / 1.1
        x = (((x + 1) * 2) - 4) / 1.1
        assert_allclose(xd.asndarray(), x)

    def test_arithmetic(self, x, xd):
        xd = xd * 2 + xd
        x = x * 2 + x
        assert_allclose(xd.asndarray(), x)

    def test_broadcast_row(self, x, xd):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xd = xd + a
        x = x + a
        assert_allclose(xd.asndarray(), x)

    def test_broadcast_col(self, x, xd):
        if sys.version_info[0] == 2 and isinstance(
            xd, zap.beam.array.ndarray_pcollection
        ):  # TODO: fix this
            return
        a = np.array([[1.0], [2.0], [3.0]])
        xd = xd + a
        x = x + a
        assert_allclose(xd.asndarray(), x)

    def test_eq(self, x, xd):
        xd = xd == 0.0
        x = x == 0.0
        assert xd.dtype == x.dtype
        assert_allclose(xd.asndarray(), x)

    def test_ne(self, x, xd):
        xd = xd != 0.0
        x = x != 0.0
        assert_allclose(xd.asndarray(), x)

    def test_invert(self, x, xd):
        xd = ~(xd == 0.0)
        x = ~(x == 0.0)
        assert_allclose(xd.asndarray(), x)

    def test_inplace(self, x, xd):
        xd += 1
        x += 1
        assert_allclose(xd.asndarray(), x)

    def test_simple_index(self, x, xd):
        xd = xd[0]
        x = x[0]
        assert_allclose(xd, x)

    def test_boolean_index(self, x, xd):
        xd = np.sum(xd, axis=1)  # sum rows
        xd = xd[xd > 5]
        x = np.sum(x, axis=1)  # sum rows
        x = x[x > 5]
        assert_allclose(xd.asndarray(), x)

    def test_subset_cols(self, x, xd):
        subset = np.array([True, False, True, False, True])
        xd = xd[:, subset]
        x = x[:, subset]
        assert xd.shape == x.shape
        assert_allclose(xd.asndarray(), x)

    def test_subset_rows(self, x, xd):
        subset = np.array([True, False, True])
        xd = xd[subset, :]
        x = x[subset, :]
        assert xd.shape == x.shape
        assert_allclose(xd.asndarray(), x)

    def test_newaxis(self, x, xd):
        xd = np.sum(xd, axis=1)[:, np.newaxis]
        x = np.sum(x, axis=1)[:, np.newaxis]
        assert_allclose(xd.asndarray(), x)

    def test_log1p(self, x, xd):
        log1pnps = np.log1p(xd).asndarray()
        log1pnp = np.log1p(x)
        assert_allclose(log1pnps, log1pnp)

    def test_sum_cols(self, x, xd):
        xd = np.sum(xd, axis=0)
        x = np.sum(x, axis=0)
        assert_allclose(xd.asndarray(), x)

    def test_sum_rows(self, x, xd):
        xd = np.sum(xd, axis=1)
        x = np.sum(x, axis=1)
        assert_allclose(xd.asndarray(), x)

    def test_mean_cols(self, x, xd):
        xd = np.mean(xd, axis=0)
        x = np.mean(x, axis=0)
        assert_allclose(xd.asndarray(), x)

    def test_mean_rows(self, x, xd):
        xd = np.mean(xd, axis=1)
        x = np.mean(x, axis=1)
        assert_allclose(xd.asndarray(), x)

    def test_var(self, x, xd):
        def var(x):
            mean = x.mean(axis=0)
            mean_sq = np.multiply(x, x).mean(axis=0)
            return mean_sq - mean ** 2

        varnps = var(xd).asndarray()
        varnp = var(x)
        assert_allclose(varnps, varnp)

    def test_write_zarr(self, x, xd, tmpdir):
        output_file_zarr = str(tmpdir.join("xd.zarr"))
        xd.to_zarr(output_file_zarr, xd.chunks)
        # read back as zarr directly and check it is the same as x
        z = zarr.open(
            output_file_zarr, mode="r", shape=x.shape, dtype=x.dtype, chunks=(2, 5)
        )
        arr = z[:]
        assert_allclose(arr, x)

    def test_zeros(self, zeros):
        totals = np.sum(zeros, axis=0)
        x = np.array([0, 0, 0, 0, 0])
        assert_allclose(totals.asndarray(), x)

    def test_ones(self, ones):
        totals = np.sum(ones, axis=0)
        x = np.array([3, 3, 3, 3, 3])
        assert_allclose(totals.asndarray(), x)
