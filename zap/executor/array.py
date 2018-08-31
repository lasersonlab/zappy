import builtins
import numpy as np
import zarr

from functools import partial

from zap.base import *  # include everything in zap.base and hence base numpy
from zap.zarr_spark import (
    get_chunk_indices,
    read_zarr_chunk,
    write_chunk,
    write_chunk_gcs,
)


def identity(x):
    return x


def to_tuple(x):
    if isinstance(x, tuple):
        return x
    return (x,)


def flatzip(a, b):
    # zip but don't nest
    return [to_tuple(x) + to_tuple(y) for (x, y) in zip(a, b)]


def compose2(f, g):
    return lambda x: f(g(x))


def binary_compose2(f, g):
    return lambda x, y: f(g(x), y)


class ndarray_executor(ndarray_dist):
    """A numpy.ndarray backed by chunked storage"""

    def __init__(
        self,
        executor,
        inputs,
        shape,
        chunks,
        dtype,
        partition_row_counts=None,
        f=identity,
    ):
        self.executor = executor
        self.inputs = inputs
        self.f = f
        self.ndim = len(shape)
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        if partition_row_counts is None:
            partition_row_counts = [chunks[0]] * (shape[0] // chunks[0])
            remaining = shape[0] % chunks[0]
            if remaining != 0:
                partition_row_counts.append(remaining)
        self.partition_row_counts = partition_row_counts

    def _new(
        self,
        inputs,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        f=identity,
    ):
        if shape is None:
            shape = self.shape
        if chunks is None:
            chunks = self.chunks
        if dtype is None:
            dtype = self.dtype
        if partition_row_counts is None:
            partition_row_counts = self.partition_row_counts
        return ndarray_executor(
            self.executor, inputs, shape, chunks, dtype, partition_row_counts, f
        )

    def _new_or_copy(
        self,
        inputs,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        copy=True,
        f=identity,
    ):
        if copy:
            return self._new(inputs, shape, chunks, dtype, partition_row_counts, f)
        else:
            self.inputs = inputs
            if shape is not None:
                self.shape = shape
            if chunks is not None:
                self.chunks = chunks
            if dtype is not None:
                self.dtype = dtype
            if partition_row_counts is not None:
                self.partition_row_counts = partition_row_counts
            if f is not None:
                self.f = f
            return self

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, executor, arr, chunks):
        shape = arr.shape
        inputs = [
            (read_zarr_chunk(arr, chunks, i),) for i in get_chunk_indices(shape, chunks)
        ]
        return cls(executor, inputs, shape, chunks, arr.dtype)
        # TODO: defer reading the Zarr chunks
        # inputs = [(i,) for i in get_chunk_indices(shape, chunks)]
        # return cls(executor, inputs, shape, chunks, arr.dtype, f=partial(read_zarr_chunk, arr, chunks))

    @classmethod
    def from_zarr(cls, executor, zarr_file):
        """
        Read a Zarr file as an ndarray_executor object.
        """
        z = zarr.open(zarr_file, mode="r")
        shape, chunks = z.shape, z.chunks
        inputs = [
            (read_zarr_chunk(z, chunks, i),) for i in get_chunk_indices(shape, chunks)
        ]
        return cls(executor, inputs, shape, chunks, z.dtype)
        # TODO: defer reading the Zarr chunks
        # inputs = [(i,) for i in get_chunk_indices(shape, chunks)]
        # return cls(executor, inputs, shape, chunks, z.dtype, f=partial(read_zarr_chunk, z, chunks))

    def _compute(self):
        def f(x):
            return self.f(*x)

        return list(self.executor.map(f, self.inputs))

    def asndarray(self):
        inputs = self._compute()
        local_row_counts = [len(arr) for arr in inputs]
        assert local_row_counts == list(self.partition_row_counts), (
            "Local row counts: %s; partition row counts: %s"
            % (local_row_counts, self.partition_row_counts)
        )
        arr = np.concatenate(inputs)
        assert arr.shape[0] == builtins.sum(self.partition_row_counts), (
            "Local #rows: %s; partition row counts total: %s"
            % (arr.shape[0], builtins.sum(self.partition_row_counts))
        )
        return arr

    def _write_zarr(self, store, chunks, write_chunk_fn):
        # partitioned_rdd = repartition_chunks(
        #     self.sc, self.rdd, chunks, self.partition_row_counts
        # )  # repartition if needed
        partitioned_inputs = self._compute()  # TODO: repartition if needed
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)

        for (idx, arr) in enumerate(partitioned_inputs):
            write_chunk_fn((idx, arr))

    def to_zarr(self, zarr_file, chunks):
        """
        Write an anndata object to a Zarr file.
        """
        self._write_zarr(zarr_file, chunks, write_chunk(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an anndata object to a Zarr file on GCS.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        self._write_zarr(
            store, chunks, write_chunk_gcs(gcs_path, gcs_project, gcs_token)
        )

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 0:  # mean of each column
            result = [(x.shape[0], np.sum(x, axis=0)) for x in self._compute()]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=0) / total_count
            inputs = [(mean,)]
            return self._new(
                inputs, mean.shape, mean.shape, partition_row_counts=mean.shape
            )
        return NotImplemented

    def sum(self, axis=None):
        if axis == 0:  # sum of each column
            result = [np.sum(x, axis=0) for x in self._compute()]
            s = np.sum(result, axis=0)
            inputs = [(s,)]
            return self._new(inputs, s.shape, s.shape, partition_row_counts=s.shape)
        elif axis == 1:  # sum of each row

            def newfunc(x):
                return np.sum(x, axis=1)

            return self._new(
                self.inputs,
                (self.shape[0],),
                (self.chunks[0],),
                f=compose2(newfunc, self.f),
            )
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        return self._new_or_copy(
            self.inputs, dtype=dtype, copy=copy, f=compose2(func, self.f)
        )

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        def newfunc(x):
            return func(x, x)

        return self._new_or_copy(
            self.inputs, dtype=dtype, copy=copy, f=compose2(newfunc, self.f)
        )

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize

        def newfunc(x):
            return func(x, other)

        return self._new_or_copy(
            self.inputs, dtype=dtype, copy=copy, f=compose2(newfunc, self.f)
        )

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        return NotImplemented

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            new_inputs = flatzip(self.inputs, other.inputs)
            return self._new_or_copy(
                new_inputs, dtype=dtype, copy=copy, f=binary_compose2(func, self.f)
            )
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        # almost identical to row subset below
        subset = item
        # materialize index ndarray_dist to ndarray
        if isinstance(subset, ndarray_executor):
            subset = subset.asndarray()
        partition_row_subsets = self._copartition(subset)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
        new_shape = (builtins.sum(new_partition_row_counts),)

        def newfunc(x, y):
            return x[y]

        return self._new(
            flatzip(self.inputs, partition_row_subsets),
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
            f=binary_compose2(newfunc, self.f),
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)

            def newfunc(x):
                return x[:, np.newaxis]

            return self._new(
                self.inputs,
                shape=new_shape,
                chunks=new_chunks,
                partition_row_counts=self.partition_row_counts,
                f=compose2(newfunc, self.f),
            )
        subset = item[1]
        # materialize index ndarray_dist to ndarray
        if isinstance(subset, ndarray_executor):
            subset = subset.asndarray()
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)

        def newfunc(x):
            return x[item]

        return self._new(
            self.inputs,
            shape=new_shape,
            chunks=new_chunks,
            partition_row_counts=self.partition_row_counts,
            f=compose2(newfunc, self.f),
        )

    def _row_subset(self, item):
        subset = item[0]
        # materialize index ndarray_dist to ndarray
        if isinstance(subset, ndarray_executor):
            subset = subset.asndarray()
        partition_row_subsets = self._copartition(subset)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])

        def newfunc(x, y):
            return x[y, :]

        return self._new(
            flatzip(self.inputs, partition_row_subsets),
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
            f=binary_compose2(newfunc, self.f),
        )

    def _copartition(self, arr):
        partition_row_subsets = np.split(
            arr, np.cumsum(self.partition_row_counts)[0:-1]
        )
        if len(partition_row_subsets[-1]) == 0:
            partition_row_subsets = partition_row_subsets[0:-1]
        return partition_row_subsets
