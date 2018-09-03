import concurrent.futures
import logging
import pytest
import sys
import zap.base as np  # zap includes everything in numpy, with some overrides and new functions
import zap.executor.array
import zap.local.array
import zap.spark.array
import zarr

from numpy.testing import assert_allclose
from pyspark.sql import SparkSession

TESTS = [0, 1, 2, 3, 6, 7]

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
            # zarr local
            yield zap.local.array.ndarray_dist_local.from_zarr(xz)
        elif request.param == 1:
            # in-memory ndarray local
            yield zap.local.array.ndarray_dist_local.from_ndarray(x.copy(), chunks)
        elif request.param == 2:
            # zarr spark
            yield zap.spark.array.array_rdd_zarr(sc, xz)
        elif request.param == 3:
            # in-memory ndarray spark
            yield zap.spark.array.array_rdd(sc, x.copy(), chunks)
        elif request.param == 4:
            # zarr beam
            pipeline_options = PipelineOptions()
            pipeline = beam.Pipeline(options=pipeline_options)
            yield zap.beam.array.ndarray_pcollection.from_zarr(pipeline, xz)
        elif request.param == 5:
            # in-memory ndarray beam
            pipeline_options = PipelineOptions()
            pipeline = beam.Pipeline(options=pipeline_options)
            yield zap.beam.array.ndarray_pcollection.from_ndarray(
                pipeline, x.copy(), chunks
            )
        elif request.param == 6:
            # zarr executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.ndarray_executor.from_zarr(executor, xz)
        elif request.param == 7:
            # in-memory ndarray executor
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                yield zap.executor.array.ndarray_executor.from_ndarray(
                    executor, x.copy(), chunks
                )

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

    def test_broadcast(self, x, xd):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
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

    def test_mean(self, x, xd):
        def mean(x):
            return x.mean(axis=0)

        meannps = mean(xd).asndarray()
        meannp = mean(x)
        assert_allclose(meannps, meannp)

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


if __name__ == "__main__":
    unittest.main()
