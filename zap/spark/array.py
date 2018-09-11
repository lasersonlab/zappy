import builtins
import numpy as np
import zarr

from zap.base import *  # include everything in zap.base and hence base numpy
from zap.zarr_spark import repartition_chunks


def array_rdd(sc, arr, chunks):
    return ndarray_rdd.from_ndarray(sc, arr, chunks)


def array_rdd_zarr(sc, zarr_file):
    return ndarray_rdd.from_zarr(sc, zarr_file)


# ndarray in Spark


class ndarray_rdd(ndarray_dist):
    """A numpy.ndarray backed by a Spark RDD"""

    def __init__(self, sc, rdd, shape, chunks, dtype, partition_row_counts=None):
        ndarray_dist.__init__(self, shape, chunks, dtype, partition_row_counts)
        self.sc = sc
        self.rdd = rdd

    def _new(self, rdd, shape=None, chunks=None, dtype=None, partition_row_counts=None):
        if shape is None:
            shape = self.shape
        if chunks is None:
            chunks = self.chunks
        if dtype is None:
            dtype = self.dtype
        if partition_row_counts is None:
            partition_row_counts = self.partition_row_counts
        return ndarray_rdd(self.sc, rdd, shape, chunks, dtype, partition_row_counts)

    def _new_or_copy(
        self,
        rdd,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        copy=True,
    ):
        if copy:
            return self._new(rdd, shape, chunks, dtype, partition_row_counts)
        else:
            self.rdd = rdd
            if shape is not None:
                self.shape = shape
            if chunks is not None:
                self.chunks = chunks
            if dtype is not None:
                self.dtype = dtype
            if partition_row_counts is not None:
                self.partition_row_counts = partition_row_counts
            return self

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, sc, arr, chunks):
        func, chunk_indices = ndarray_dist._read_chunks(arr, chunks)
        rdd = sc.parallelize(chunk_indices, len(chunk_indices)).map(func)
        return cls(sc, rdd, arr.shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        """
        Read a Zarr file as an ndarray_rdd object.
        """
        arr = zarr.open(zarr_file, mode="r")
        return cls.from_ndarray(sc, arr, arr.chunks)

    def _compute(self):
        return self.rdd.collect()

    def _repartition_chunks(self, chunks):
        partitioned_rdd = repartition_chunks(self.sc, self.rdd, chunks, self.partition_row_counts)
        partition_row_counts = [chunks[0]] * (self.shape[0] // chunks[0])
        remaining = self.shape[0] % chunks[0]
        if remaining != 0:
            partition_row_counts.append(remaining)
        return self._new_or_copy(partitioned_rdd, chunks=chunks, partition_row_counts=partition_row_counts, copy=True)

    def _write_zarr(self, store, chunks, write_chunk_fn):
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)

        def index_partitions(index, iterator):
            values = list(iterator)
            assert len(values) == 1  # 1 numpy array per partition
            return [(index, values[0])]

        self.rdd.mapPartitionsWithIndex(index_partitions).foreach(write_chunk_fn)

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 0:  # mean of each column
            result = self.rdd.map(lambda x: (x.shape[0], np.sum(x, axis=0))).collect()
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=0) / total_count
            rdd = self.rdd.ctx.parallelize([mean])
            return self._new(
                rdd, mean.shape, mean.shape, partition_row_counts=mean.shape
            )
        return NotImplemented

    def sum(self, axis=None):
        if axis == 0:  # sum of each column
            result = self.rdd.map(lambda x: np.sum(x, axis=0)).collect()
            s = np.sum(result, axis=0)
            rdd = self.rdd.ctx.parallelize([s])
            return self._new(rdd, s.shape, s.shape, partition_row_counts=s.shape)
        elif axis == 1:  # sum of each row
            return self._new(
                self.rdd.map(lambda x: np.sum(x, axis=1)),
                (self.shape[0],),
                (self.chunks[0],),
            )
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        new_rdd = self.rdd.map(lambda x: func(x))
        return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        new_rdd = self.rdd.map(lambda x: func(x, x))
        return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        # TODO: should send 'other' as a Spark broadcast
        new_rdd = self.rdd.map(lambda x: func(x, other))
        return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        other = asarray(other)  # materialize
        partition_row_subsets = self._copartition(other, self.partition_row_counts)
        repartitioned_other_rdd = self.sc.parallelize(
            partition_row_subsets, len(partition_row_subsets)
        )
        new_rdd = self.rdd.zip(repartitioned_other_rdd).map(lambda p: func(p[0], p[1]))
        return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            new_rdd = self.rdd.zip(other.rdd).map(lambda p: func(p[0], p[1]))
            return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)
        elif other.shape[1] == 1:
            partition_row_subsets = self._copartition(
                other.asndarray(), self.partition_row_counts
            )
            repartitioned_other_rdd = self.sc.parallelize(
                partition_row_subsets, len(partition_row_subsets)
            )
            new_rdd = self.rdd.zip(repartitioned_other_rdd).map(
                lambda p: func(p[0], p[1])
            )
            return self._new_or_copy(new_rdd, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        subset = asarray(item)  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts),)
        # leave new chunks undefined since they are not necessarily equal-sized
        subset_rdd = self.sc.parallelize(
            partition_row_subsets, len(partition_row_subsets)
        )
        return self._new(
            self.rdd.zip(subset_rdd).map(lambda p: p[0][p[1]]),
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            return self._new(
                self.rdd.map(lambda x: x[:, np.newaxis]),
                shape=new_shape,
                chunks=new_chunks,
                partition_row_counts=self.partition_row_counts,
            )
        subset = asarray(item[1])  # materialize
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        return self._new(
            self.rdd.map(lambda x: x[item]),
            shape=new_shape,
            chunks=new_chunks,
            partition_row_counts=self.partition_row_counts,
        )

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
        # leave new chunks undefined since they are not necessarily equal-sized
        subset_rdd = self.sc.parallelize(
            partition_row_subsets, len(partition_row_subsets)
        )
        return self._new(
            self.rdd.zip(subset_rdd).map(lambda p: p[0][p[1], :]),
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )
