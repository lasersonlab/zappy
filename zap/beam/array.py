import apache_beam as beam
import builtins
import os
import numpy as np
import shutil
import tempfile
import zarr

from apache_beam.pvalue import AsDict
from zap.base import *  # include everything in zap.base and hence base numpy

sym_counter = 0


def gensym(name):
    global sym_counter
    sym_counter += 1
    return "%s_%s" % (name, sym_counter)


# ndarray in Beam


class ndarray_pcollection(ndarray_dist):
    """A numpy.ndarray backed by a Beam PCollection"""

    def __init__(
        self,
        pipeline,
        pcollection,
        shape,
        chunks,
        dtype,
        partition_row_counts=None,
        tmp_dir=None,
    ):
        ndarray_dist.__init__(self, shape, chunks, dtype, partition_row_counts)
        self.pipeline = pipeline
        self.pcollection = pcollection
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp("-asndarray", "beam-")
        self.tmp_dir = tmp_dir

    def close(self):
        shutil.rmtree(self.tmp_dir)

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, pipeline, arr, chunks):
        func, chunk_indices = ndarray_dist._read_chunks(arr, chunks)
        # use the first component of chunk index as an index (assumes rows are one chunk wide)
        pcollection = (
            pipeline
            | beam.Create(chunk_indices)
            | beam.Map(lambda chunk_index: (chunk_index[0], func(chunk_index)))
        )
        return cls(pipeline, pcollection, arr.shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, pipeline, zarr_file):
        """
        Read a Zarr file as an ndarray_pcollection object.
        """
        arr = zarr.open(zarr_file, mode="r")
        return cls.from_ndarray(pipeline, arr, arr.chunks)

    def _compute(self):
        # create a temporary subdirectory to materialize arrays to
        sym = gensym("asndarray")
        subdir = "%s/%s" % (self.tmp_dir, sym)
        os.mkdir(subdir)

        # save files
        def save(indexed_row):
            index, row = indexed_row
            with open("%s/%s" % (subdir, index), "w") as file:
                np.save(file, row)

        self.pcollection | sym >> beam.Map(save)
        result = self.pipeline.run()
        result.wait_until_finish()

        # read back files
        local_rows = [None] * len(self.partition_row_counts)
        for filename in os.listdir(subdir):
            index = int(filename)
            row = np.load(os.path.join(subdir, filename))
            local_rows[index] = row

        return local_rows

    def _write_zarr(self, store, chunks, write_chunk_fn):
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)
        self.pcollection | gensym("write_zarr") >> beam.Map(write_chunk_fn)

        result = self.pipeline.run()
        result.wait_until_finish()

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def sum(a, axis=None):
        if axis == 0:  # sum of each column
            # note that unlike in the Spark implementation, nothing is materialized here - the whole computation is deferred
            # this should make things faster
            new_pcollection = (
                a.pcollection
                | gensym("sum")
                >> beam.Map(lambda pair: np.sum(pair[1], axis=0))  # drop indexes
                | gensym("sum_combine")
                >> beam.CombineGlobally(lambda values: np.sum(values, axis=0))
                | gensym("add_index") >> beam.Map(lambda elt: (0, elt))
            )  # only one row
            new_shape = (a.shape[1],)
            return a._new(
                pcollection=new_pcollection,
                shape=new_shape,
                chunks=new_shape,
                partition_row_counts=new_shape,
            )
        elif axis == 1:  # sum of each row
            new_pcollection = a.pcollection | gensym("sum") >> beam.Map(
                lambda pair: (pair[0], np.sum(pair[1], axis=1))
            )
            return a._new(
                pcollection=new_pcollection, shape=(a.shape[0],), chunks=(a.chunks[0],)
            )
        return NotImplemented

    def mean(a, axis=None):
        class CountAndSumsFn(beam.DoFn):
            def process(self, element):
                (idx, row) = element
                count = row.shape[0]
                sum = np.sum(row, axis=0)
                return [(count, sum)]

        class MeanColsFn(beam.CombineFn):
            def __init__(self, num_cols):
                self.num_cols = num_cols

            def create_accumulator(self):
                return 0.0, np.zeros(self.num_cols)

            def add_input(self, count_sum, input):
                (count, sum) = count_sum
                (input_count, input_sum) = input
                return count + input_count, np.add(sum, input_sum)

            def merge_accumulators(self, accumulators):
                counts, sums = zip(*accumulators)
                return builtins.sum(counts), np.sum(sums, axis=0)

            def extract_output(self, count_sum):
                (count, sum) = count_sum
                return (
                    0,
                    sum / count if count else float("NaN"),
                )  # one element with index 0

        if axis == 0:  # mean of each column
            # note that unlike in the Spark implementation, nothing is materialized here - the whole computation is deferred
            # this should make things faster
            new_pcollection = (
                a.pcollection
                | gensym("count_and_sum") >> beam.ParDo(CountAndSumsFn())
                | gensym("mean_cols") >> beam.CombineGlobally(MeanColsFn(a.shape[1]))
            )
            new_shape = (a.shape[1],)
            return a._new(
                pcollection=new_pcollection,
                shape=new_shape,
                chunks=new_shape,
                partition_row_counts=new_shape,
            )
        return NotImplemented

    # TODO: more calculation methods here

    # TODO: for Beam we should be able to avoid materializing everything - defer all the computations (even shapes, row partitions, etc)!

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1]))
        )
        return self._new(pcollection=new_pcollection, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1], pair[1]))
        )
        return self._new(pcollection=new_pcollection, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        # TODO: should send 'other' as a Beam side input
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1], other))
        )
        return self._new(pcollection=new_pcollection, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        # TODO: Beam (side input)
        return NotImplemented

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        if self.partition_row_counts == other.partition_row_counts:
            # args have the same rows (and partitioning) so use zip to combine then apply the operator
            # Beam doesn't have a direct equivalent of Spark's zip function, so we use CoGroupByKey (is this less efficient?)
            def combine_indexed_dict(indexed_dict):
                idx, dict = indexed_dict
                return idx, func(dict["self"][0], dict["other"][0])

            new_pcollection = (
                {"self": self.pcollection, "other": other.pcollection}
                | beam.CoGroupByKey()
                | gensym(func.__name__) >> beam.Map(combine_indexed_dict)
            )
            return self._new(pcollection=new_pcollection, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        subset = asarray(item)  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts),)

        # Beam doesn't have a direct equivalent of Spark's zip function, so we use a side input and join here
        # See https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/snippets/snippets.py#L1295
        subset_pcollection = self.pipeline | gensym(
            "partition_row_subsets"
        ) >> beam.Create(enumerate(partition_row_subsets))

        def join_row_with_subset(index_row, subset_dict):
            index, row = index_row
            return index, row[subset_dict[index]]

        new_pcollection = self.pcollection | gensym("row_subset") >> beam.Map(
            join_row_with_subset, AsDict(subset_pcollection)
        )

        return self._new(
            pcollection=new_pcollection,
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )

    def _column_subset(self, item):
        if item[1] is np.newaxis:  # add new col axis
            new_num_cols = 1
            new_shape = (self.shape[0], new_num_cols)
            new_chunks = (self.chunks[0], new_num_cols)
            new_pcollection = self.pcollection | gensym(
                "column_subset_newaxis"
            ) >> beam.Map(lambda pair: (pair[0], pair[1][:, np.newaxis]))
            return self._new(
                pcollection=new_pcollection, shape=new_shape, chunks=new_chunks
            )
        subset = asarray(item[1])  # materialize
        new_pcollection = self.pcollection | gensym("column_subset") >> beam.Map(
            lambda pair: (pair[0], pair[1][item])
        )
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        return self._new(
            pcollection=new_pcollection, shape=new_shape, chunks=new_chunks
        )

    def _row_subset(self, item):
        subset = asarray(item[0])  # materialize
        partition_row_subsets = self._copartition(subset, self.partition_row_counts)
        new_partition_row_counts = self._partition_row_counts(partition_row_subsets)
        new_shape = (builtins.sum(new_partition_row_counts), self.shape[1])

        # Beam doesn't have a direct equivalent of Spark's zip function, so we use a side input and join here
        # See https://github.com/apache/beam/blob/master/sdks/python/apache_beam/examples/snippets/snippets.py#L1295
        subset_pcollection = self.pipeline | gensym(
            "partition_row_subsets"
        ) >> beam.Create(enumerate(partition_row_subsets))

        def join_row_with_subset(index_row, subset_dict):
            index, row = index_row
            return index, row[subset_dict[index], :]

        new_pcollection = self.pcollection | gensym("row_subset") >> beam.Map(
            join_row_with_subset, AsDict(subset_pcollection)
        )

        return self._new(
            pcollection=new_pcollection,
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )
