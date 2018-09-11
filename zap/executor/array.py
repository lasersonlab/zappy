import builtins
import numpy as np
import zarr

from zap.base import *  # include everything in zap.base and hence base numpy
from zap.executor.dag import DAG


class ndarray_executor(ndarray_dist):
    """A numpy.ndarray backed by chunked storage"""

    # new dag
    def __init__(
        self, executor, dag, input, shape, chunks, dtype, partition_row_counts=None
    ):
        ndarray_dist.__init__(self, shape, chunks, dtype, partition_row_counts)
        self.executor = executor
        self.dag = dag
        self.input = input

    # same dag
    def _new(
        self, input, shape=None, chunks=None, dtype=None, partition_row_counts=None
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
            self.executor, self.dag, input, shape, chunks, dtype, partition_row_counts
        )

    # same dag
    def _new_or_copy(
        self,
        input,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        copy=True,
    ):
        if copy:
            return self._new(input, shape, chunks, dtype, partition_row_counts)
        else:
            self.input = input
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
    def from_ndarray(cls, executor, arr, chunks):
        func, chunk_indices = ndarray_dist._read_chunks(arr, chunks)
        dag = DAG(executor)
        # the input is just the chunk indices
        input = dag.add_input(chunk_indices)
        # add a transform to read chunks
        input = dag.transform(func, [input])
        return cls(executor, dag, input, arr.shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, executor, zarr_file):
        """
        Read a Zarr file as an ndarray_executor object.
        """
        arr = zarr.open(zarr_file, mode="r")
        return cls.from_ndarray(executor, arr, arr.chunks)

    def _compute(self):
        return list(self.dag.compute(self.input))

    def _write_zarr(self, store, chunks, write_chunk_fn):
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)
        indices = self.dag.add_input(list(range(len(self.partition_row_counts))))
        output = self.dag.transform(
            lambda x, y: write_chunk_fn((x, y)), [indices, self.input]
        )
        list(self.dag.compute(output))

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 0:  # mean of each column
            result = [(x.shape[0], np.sum(x, axis=0)) for x in self._compute()]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=0) / total_count
            # new dag
            dag = DAG(self.executor)
            partitioned_input = [mean]
            input = dag.add_input(partitioned_input)
            return ndarray_executor(
                self.executor,
                dag,
                input,
                mean.shape,
                mean.shape,
                self.dtype,
                partition_row_counts=mean.shape,
            )
        return NotImplemented

    def sum(self, axis=None):
        if axis == 0:  # sum of each column
            result = [np.sum(x, axis=0) for x in self._compute()]
            s = np.sum(result, axis=0)
            # new dag
            dag = DAG(self.executor)
            partitioned_input = [s]
            input = dag.add_input(partitioned_input)
            return ndarray_executor(
                self.executor,
                dag,
                input,
                s.shape,
                s.shape,
                self.dtype,
                partition_row_counts=s.shape,
            )
        elif axis == 1:  # sum of each row
            input = self.dag.transform(lambda x: np.sum(x, axis=1), [self.input])
            return self._new(input, (self.shape[0],), (self.chunks[0],))
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        input = self.dag.transform(func, [self.input])
        return self._new_or_copy(input, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        input = self.dag.transform(lambda x: func(x, x), [self.input])
        return self._new_or_copy(input, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        input = self.dag.transform(lambda x: func(x, other), [self.input])
        return self._new_or_copy(input, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        other = asarray(other)  # materialize
        partition_row_subsets = self._copartition(other, self.partition_row_counts)
        side_input = self.dag.add_input(partition_row_subsets)
        input = self.dag.transform(func, [self.input, side_input])
        return self._new_or_copy(input, dtype=dtype, copy=copy)

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            input = self.dag.transform(func, [self.input, other.input])
            return self._new_or_copy(input, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        # almost identical to row subset below (only lambda has different indexing)
        subset = asarray(item)  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts),)
        side_input = self.dag.add_input(partition_row_subsets)
        input = self.dag.transform(lambda x, y: x[y], [self.input, side_input])
        return self._new(
            input, shape=new_shape, partition_row_counts=new_partition_row_counts
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            input = self.dag.transform(lambda x: x[:, np.newaxis], [self.input])
            return self._new(
                input,
                shape=new_shape,
                chunks=new_chunks,
                partition_row_counts=self.partition_row_counts,
            )
        subset = asarray(item[1])  # materialize
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        input = self.dag.transform(lambda x: x[item], [self.input])
        return self._new(
            input,
            shape=new_shape,
            chunks=new_chunks,
            partition_row_counts=self.partition_row_counts,
        )

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
        side_input = self.dag.add_input(partition_row_subsets)
        input = self.dag.transform(lambda x, y: x[y, :], [self.input, side_input])
        return self._new(
            input, shape=new_shape, partition_row_counts=new_partition_row_counts
        )
