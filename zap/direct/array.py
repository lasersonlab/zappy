import builtins
import numpy as np
import zarr

from zap.base import *  # include everything in zap.base and hence base numpy


class ndarray_dist_direct(ndarray_dist):
    """A numpy.ndarray backed by chunked storage"""

    def __init__(self, local_rows, shape, chunks, dtype, partition_row_counts=None):
        self.local_rows = local_rows
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
        self, local_rows, shape=None, chunks=None, dtype=None, partition_row_counts=None
    ):
        if shape is None:
            shape = self.shape
        if chunks is None:
            chunks = self.chunks
        if dtype is None:
            dtype = self.dtype
        if partition_row_counts is None:
            partition_row_counts = self.partition_row_counts
        return ndarray_dist_direct(
            local_rows, shape, chunks, dtype, partition_row_counts
        )

    def _new_or_copy(
        self,
        local_rows,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        copy=True,
    ):
        if copy:
            return self._new(local_rows, shape, chunks, dtype, partition_row_counts)
        else:
            self.local_rows = local_rows
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

    def _compute(self):
        return self.local_rows

    def _get_partition_row_counts(self):
        return self.partition_row_counts

    def _write_zarr(self, store, chunks, write_chunk_fn):
        # partitioned_rdd = repartition_chunks(
        #     self.sc, self.rdd, chunks, self.partition_row_counts
        # )  # repartition if needed
        partitioned_local_rows = self.local_rows  # TODO: repartition if needed
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)

        for (idx, arr) in enumerate(partitioned_local_rows):
            write_chunk_fn((idx, arr))

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 0:  # mean of each column
            result = [(x.shape[0], np.sum(x, axis=0)) for x in self.local_rows]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=0) / total_count
            local_rows = [mean]
            return self._new(
                local_rows, mean.shape, mean.shape, partition_row_counts=mean.shape
            )
        return NotImplemented

    def sum(self, axis=None):
        if axis == 0:  # sum of each column
            result = [np.sum(x, axis=0) for x in self.local_rows]
            s = np.sum(result, axis=0)
            local_rows = [s]
            return self._new(local_rows, s.shape, s.shape, partition_row_counts=s.shape)
        elif axis == 1:  # sum of each row
            return self._new(
                [np.sum(x, axis=1) for x in self.local_rows],
                (self.shape[0],),
                (self.chunks[0],),
            )
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        new_local_rows = [func(x) for x in self.local_rows]
        return self._new_or_copy(new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        new_local_rows = [func(x, x) for x in self.local_rows]
        return self._new_or_copy(new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        new_local_rows = [func(x, other) for x in self.local_rows]
        return self._new_or_copy(new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        other = asarray(other)  # materialize
        partition_row_subsets = self._copartition(other, self.partition_row_counts)
        new_local_rows = [
            func(p[0], p[1]) for p in zip(self.local_rows, partition_row_subsets)
        ]
        return self._new_or_copy(new_local_rows, dtype=dtype, copy=copy)

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            new_local_rows = [
                func(p[0], p[1]) for p in zip(self.local_rows, other.local_rows)
            ]
            return self._new_or_copy(new_local_rows, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        # almost identical to row subset below
        subset = asarray(item)  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
        new_shape = (builtins.sum(new_partition_row_counts),)
        return self._new(
            [p[0][p[1]] for p in zip(self.local_rows, partition_row_subsets)],
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            return self._new(
                [x[:, np.newaxis] for x in self.local_rows],
                shape=new_shape,
                chunks=new_chunks,
                partition_row_counts=self.partition_row_counts,
            )
        subset = asarray(item[1])  # materialize
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        return self._new(
            [x[item] for x in self.local_rows],
            shape=new_shape,
            chunks=new_chunks,
            partition_row_counts=self.partition_row_counts,
        )

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
        return self._new(
            [p[0][p[1], :] for p in zip(self.local_rows, partition_row_subsets)],
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )
