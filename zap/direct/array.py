import builtins
import numpy as np
import zarr

from zap.base import *  # include everything in zap.base and hence base numpy
from zap.zarr_util import get_chunk_sizes


def from_ndarray(arr, chunks):
    return ndarray_dist_direct.from_ndarray(arr, chunks)


def from_zarr(zarr_file):
    return ndarray_dist_direct.from_zarr(zarr_file)


def zeros(shape, chunks, dtype=float):
    return ndarray_dist_direct.zeros(shape, chunks, dtype)


def ones(shape, chunks, dtype=float):
    return ndarray_dist_direct.ones(shape, chunks, dtype)


class ndarray_dist_direct(ndarray_dist):
    """A numpy.ndarray backed by chunked storage"""

    def __init__(self, local_rows, shape, chunks, dtype, partition_row_counts=None):
        ndarray_dist.__init__(self, shape, chunks, dtype, partition_row_counts)
        self.local_rows = local_rows

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, arr, chunks):
        func, chunk_indices = ndarray_dist._read_chunks(arr, chunks)
        local_rows = [func(i) for i in chunk_indices]
        return cls(local_rows, arr.shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, zarr_file):
        """
        Read a Zarr file as an ndarray_dist_direct object.
        """
        arr = zarr.open(zarr_file, mode="r")
        return cls.from_ndarray(arr, arr.chunks)

    @classmethod
    def zeros(cls, shape, chunks, dtype=float):
        local_rows = [
            np.zeros(chunk, dtype=dtype) for chunk in get_chunk_sizes(shape, chunks)
        ]
        return cls(local_rows, shape, chunks, dtype)

    @classmethod
    def ones(cls, shape, chunks, dtype=float):
        local_rows = [
            np.ones(chunk, dtype=dtype) for chunk in get_chunk_sizes(shape, chunks)
        ]
        return cls(local_rows, shape, chunks, dtype)

    def _compute(self):
        return self.local_rows

    def _repartition_chunks(self, chunks):
        arr = np.concatenate(self.local_rows)
        partition_row_counts = [chunks[0]] * (self.shape[0] // chunks[0])
        remaining = self.shape[0] % chunks[0]
        if remaining != 0:
            partition_row_counts.append(remaining)
        return self._new(
            local_rows=self._copartition(arr, partition_row_counts),
            chunks=chunks,
            partition_row_counts=partition_row_counts,
        )

    def _write_zarr(self, store, chunks, write_chunk_fn):
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)

        for (idx, arr) in enumerate(self.local_rows):
            write_chunk_fn((idx, arr))

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 0:  # mean of each column
            result = [(x.shape[0], np.sum(x, axis=0)) for x in self.local_rows]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=0) / total_count
            local_rows = [mean]
            return self._new(
                local_rows=local_rows,
                shape=mean.shape,
                chunks=mean.shape,
                partition_row_counts=mean.shape,
            )
        elif axis == 1:
            return self._calc_func_axis_rowwise(np.mean, axis)
        return NotImplemented

    def _calc_func_axis_rowwise(self, func, axis):
        return self._new(
            local_rows=[func(x, axis=axis) for x in self.local_rows],
            shape=(self.shape[0],),
            chunks=(self.chunks[0],),
        )

    def _calc_func_axis_distributive(self, func, axis):
        if axis == 0:  # column-wise
            per_chunk_result = [func(x, axis=0) for x in self.local_rows]
            result = func(per_chunk_result, axis=0)
            local_rows = [result]
            return self._new(
                local_rows=local_rows,
                shape=result.shape,
                chunks=result.shape,
                partition_row_counts=result.shape,
            )
        return NotImplemented

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        new_local_rows = [func(x) for x in self.local_rows]
        return self._new(local_rows=new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        new_local_rows = [func(x, x) for x in self.local_rows]
        return self._new(local_rows=new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        new_local_rows = [func(x, other) for x in self.local_rows]
        return self._new(local_rows=new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        other = asarray(other)  # materialize
        partition_row_subsets = self._copartition(other, self.partition_row_counts)
        new_local_rows = [
            func(p[0], p[1]) for p in zip(self.local_rows, partition_row_subsets)
        ]
        return self._new(local_rows=new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            new_local_rows = [
                func(p[0], p[1]) for p in zip(self.local_rows, other.local_rows)
            ]
            return self._new(local_rows=new_local_rows, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        # almost identical to row subset below
        subset = asarray(item)  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts),)
        return self._new(
            local_rows=[
                p[0][p[1]] for p in zip(self.local_rows, partition_row_subsets)
            ],
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            return self._new(
                local_rows=[x[:, np.newaxis] for x in self.local_rows],
                shape=new_shape,
                chunks=new_chunks,
            )
        subset = asarray(item[1])  # materialize
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        return self._new(
            local_rows=[x[item] for x in self.local_rows],
            shape=new_shape,
            chunks=new_chunks,
        )

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
        return self._new(
            local_rows=[
                p[0][p[1], :] for p in zip(self.local_rows, partition_row_subsets)
            ],
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )
