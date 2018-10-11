import builtins
import datetime
import os
import pickle
import uuid

import numpy as np
import zarr

from zappy.base import *  # include everything in zappy.base and hence base numpy
from zappy.executor.dag import DAG
from zappy.zarr_util import (
    calculate_partition_boundaries,
    extract_partial_chunks,
    get_chunk_sizes,
)


def from_ndarray(executor, arr, chunks, intermediate_store=None):
    return ExecutorZappyArray.from_ndarray(executor, arr, chunks, intermediate_store)


def from_zarr(executor, zarr_file, intermediate_store=None):
    return ExecutorZappyArray.from_zarr(executor, zarr_file, intermediate_store)


def zeros(executor, shape, chunks, dtype=float, intermediate_store=None):
    return ExecutorZappyArray.zeros(executor, shape, chunks, dtype, intermediate_store)


def ones(executor, shape, chunks, dtype=float, intermediate_store=None):
    return ExecutorZappyArray.ones(executor, shape, chunks, dtype, intermediate_store)


class PywrenExecutor(object):
    """Small wrapper to make a Pywren executor behave like a concurrent.futures.Executor."""

    def __init__(
        self, pywren_executor=None, exclude_modules=None, record_job_history=True
    ):
        import pywren

        self.pywren_executor = (
            pywren_executor
            if pywren_executor is not None
            else pywren.default_executor()
        )
        self.exclude_modules = exclude_modules
        self.record_job_history = record_job_history

    def map(self, func, iterables):
        import pywren

        futures = self.pywren_executor.map(
            func, iterables, exclude_modules=self.exclude_modules
        )
        pywren.wait(futures, return_when=pywren.ALL_COMPLETED)
        # Set throw_except to False, since when writing Zarr we don't return anything.
        # However, Pywren should be improved to distinguish between no return value and an exception.
        results = [f.result(throw_except=False) for f in futures]
        if self.record_job_history:
            run_statuses = [f.run_status for f in futures]
            invoke_statuses = [f.invoke_status for f in futures]
            outdict = {
                "futures": futures,
                "run_statuses": run_statuses,
                "invoke_statuses": invoke_statuses,
            }
            logs_dir = os.path.expanduser("~/.zappy/logs")
            os.makedirs(logs_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
            filename = os.path.join(logs_dir, "pywren-{}.pickle".format(timestamp))
            with open(filename, "wb") as file:
                pickle.dump(outdict, file)
        return results


class ExecutorZappyArray(ZappyArray):
    """A numpy.ndarray backed by chunked storage"""

    def __init__(
        self,
        executor,
        dag,
        input,
        shape,
        chunks,
        dtype,
        partition_row_counts=None,
        intermediate_store=None,
    ):
        ZappyArray.__init__(self, shape, chunks, dtype, partition_row_counts)
        self.executor = executor
        self.dag = dag
        self.input = input
        if intermediate_store == None:
            intermediate_store = zarr.group()
        self.intermediate_store = intermediate_store

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, executor, arr, chunks, intermediate_store=None):
        func, chunk_indices = ZappyArray._read_chunks(arr, chunks)
        dag = DAG(executor)
        # the input is just the chunk indices
        input = dag.add_input(chunk_indices)
        # add a transform to read chunks
        input = dag.transform(func, [input])
        return cls(
            executor,
            dag,
            input,
            arr.shape,
            chunks,
            arr.dtype,
            intermediate_store=intermediate_store,
        )

    @classmethod
    def from_zarr(cls, executor, zarr_file, intermediate_store=None):
        """
        Read a Zarr file as an ExecutorZappyArray object.
        """
        arr = zarr.open(zarr_file, mode="r")
        return cls.from_ndarray(executor, arr, arr.chunks, intermediate_store)

    @classmethod
    def zeros(cls, executor, shape, chunks, dtype=float, intermediate_store=None):
        dag = DAG(executor)
        input = dag.add_input(list(get_chunk_sizes(shape, chunks)))
        input = dag.transform(lambda chunk: np.zeros(chunk, dtype=dtype), [input])
        return cls(
            executor,
            dag,
            input,
            shape,
            chunks,
            dtype,
            intermediate_store=intermediate_store,
        )

    @classmethod
    def ones(cls, executor, shape, chunks, dtype=float, intermediate_store=None):
        dag = DAG(executor)
        input = dag.add_input(list(get_chunk_sizes(shape, chunks)))
        input = dag.transform(lambda chunk: np.ones(chunk, dtype=dtype), [input])
        return cls(
            executor,
            dag,
            input,
            shape,
            chunks,
            dtype,
            intermediate_store=intermediate_store,
        )

    def _compute(self):
        return list(self.dag.compute(self.input))

    def _repartition_chunks(self, chunks):
        c = chunks[0]
        partition_row_ranges, total_rows, new_num_partitions = calculate_partition_boundaries(
            chunks, self.partition_row_counts
        )

        # make a new zarr group in the intermediate store with a unique name
        root = self.intermediate_store.create_group(str(uuid.uuid4()))
        # make a zarr group for each partition
        for index in range(new_num_partitions):
            root.create_group(str(index))

        def tmp_store(pairs):
            for pair in pairs:
                index, offsets, partial_chunk = pair[0], pair[1][0], pair[1][1]
                g = root.require_group(str(index))
                g.array("%s-%s" % (offsets[0], offsets[1]), partial_chunk, chunks=False)

        x1 = self.dag.add_input(partition_row_ranges)
        x2 = self.dag.transform(
            lambda x, y: extract_partial_chunks((x, y), chunks), [x1, self.input]
        )

        x3 = self.dag.transform(tmp_store, [x2])

        # run computation to save partial chunks
        list(self.dag.compute(x3))

        # create a new computation to read partial chunks
        def tmp_load(new_index):
            # last chunk has fewer than c rows
            if new_index == new_num_partitions - 1 and total_rows % c != 0:
                last_chunk_rows = total_rows % c
                arr = np.zeros((last_chunk_rows, chunks[1]))
            else:
                arr = np.zeros(chunks)
            g = root.require_group(str(new_index))
            for (name, partial_chunk) in g.arrays():
                new_start_offset, new_end_offset = [int(n) for n in name.split("-")]
                arr[new_start_offset:new_end_offset] = partial_chunk
            return arr

        dag = DAG(self.executor)
        input = dag.add_input(list(range(new_num_partitions)))
        input = dag.transform(tmp_load, [input])

        # TODO: delete intermediate store when dag is computed
        return ExecutorZappyArray(
            self.executor, dag, input, self.shape, chunks, self.dtype
        )

    def _write_zarr(self, store, chunks, write_chunk_fn):
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)
        indices = self.dag.add_input(list(range(len(self.partition_row_counts))))
        output = self.dag.transform(
            lambda x, y: write_chunk_fn((x, y)), [indices, self.input]
        )
        list(self.dag.compute(output))

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def _calc_mean(self, axis=None):
        if axis is None:
            result = [
                (x.shape[0] * x.shape[1], np.sum(x, axis=axis)) for x in self._compute()
            ]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=axis) / total_count
            return mean
        elif axis == 0:  # mean of each column
            result = [(x.shape[0], np.sum(x, axis=axis)) for x in self._compute()]
            total_count = builtins.sum([res[0] for res in result])
            mean = np.sum([res[1] for res in result], axis=axis) / total_count
            # new dag
            dag = DAG(self.executor)
            partitioned_input = [mean]
            input = dag.add_input(partitioned_input)
            return self._new(
                dag=dag,
                input=input,
                shape=mean.shape,
                chunks=mean.shape,
                partition_row_counts=mean.shape,
            )
        return NotImplemented

    def _calc_func_axis_rowwise(self, func, axis):
        input = self.dag.transform(lambda x: func(x, axis=axis), [self.input])
        return self._new(input=input, shape=(self.shape[0],), chunks=(self.chunks[0],))

    def _calc_func_axis_distributive(self, func, axis):
        per_chunk_result = [func(x, axis=axis) for x in self._compute()]
        result = func(per_chunk_result, axis=axis)
        if axis is None:
            return result
        elif axis == 0:  # column-wise
            # new dag
            dag = DAG(self.executor)
            partitioned_input = [result]
            input = dag.add_input(partitioned_input)
            return self._new(
                dag=dag,
                input=input,
                shape=result.shape,
                chunks=result.shape,
                partition_row_counts=result.shape,
            )
        return NotImplemented

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, out=None, dtype=None):
        input = self.dag.transform(func, [self.input])
        return self._new(input=input, out=out, dtype=dtype)

    def _binary_ufunc_self(self, func, out=None, dtype=None):
        input = self.dag.transform(lambda x: func(x, x), [self.input])
        return self._new(input=input, out=out, dtype=dtype)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, out=None, dtype=None
    ):
        other = asarray(other)  # materialize
        input = self.dag.transform(lambda x: func(x, other), [self.input])
        return self._new(input=input, out=out, dtype=dtype)

    def _binary_ufunc_broadcast_single_column(self, func, other, out=None, dtype=None):
        other = asarray(other)  # materialize
        partition_row_subsets = self._copartition(other, self.partition_row_counts)
        side_input = self.dag.add_input(partition_row_subsets)
        input = self.dag.transform(func, [self.input, side_input])
        return self._new(input=input, out=out, dtype=dtype)

    def _binary_ufunc_same_shape(self, func, other, out=None, dtype=None):
        if self.partition_row_counts == other.partition_row_counts:
            input = self.dag.transform(func, [self.input, other.input])
            return self._new(input=input, out=out, dtype=dtype)
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
            input=input, shape=new_shape, partition_row_counts=new_partition_row_counts
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            input = self.dag.transform(lambda x: x[:, np.newaxis], [self.input])
            return self._new(input=input, shape=new_shape, chunks=new_chunks)
        subset = self._materialize_index(item[1])
        new_num_cols = self._compute_dim(self.shape[1], subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        input = self.dag.transform(lambda x: x[item], [self.input])
        return self._new(input=input, shape=new_shape, chunks=new_chunks)

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])
        side_input = self.dag.add_input(partition_row_subsets)
        input = self.dag.transform(lambda x, y: x[y, :], [self.input, side_input])
        return self._new(
            input=input, shape=new_shape, partition_row_counts=new_partition_row_counts
        )
