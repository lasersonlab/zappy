import apache_beam as beam
import builtins
import os
import numpy as np
import shutil
import tempfile
import zarr

from apache_beam.pvalue import AsDict
from zap.base import *  # include everything in zap.base and hence base numpy
from zap.zarr_spark import get_chunk_indices, read_zarr_chunk

sym_counter = 0


def gensym(name):
    global sym_counter
    sym_counter += 1
    return "%s_%s" % (name, sym_counter)


# ndarray in Beam


def _read_chunk_from_arr(arr, chunks, chunk_index):
    return arr[
        chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1),
        chunks[1] * chunk_index[1] : chunks[1] * (chunk_index[1] + 1),
    ]


def _read_chunk(arr, chunks):
    """
    Return a function to read a chunk by coordinates from the given ndarray.
    """

    def read_one_chunk(chunk_index):
        return _read_chunk_from_arr(arr, chunks, chunk_index)

    return read_one_chunk


def _read_chunk_zarr(zarr_file, chunks):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        z = zarr.open(zarr_file, mode="r")
        return read_zarr_chunk(z, chunks, chunk_index)

    return read_one_chunk


def _write_chunk_zarr(zarr_file):
    """
    Return a function to write a chunk by index to the given file.
    """

    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        index, arr = index_arr
        z = zarr.open(zarr_file, mode="r+")
        chunk_size = z.chunks
        z[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


def _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token):
    """
    Return a function to write a chunk by index to the given file.
    """

    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        index, arr = index_arr
        z = zarr.open(store, mode="r+")
        chunk_size = z.chunks
        z[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


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
        self.pipeline = pipeline
        self.pcollection = pcollection
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
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp("-asndarray", "beam-")
        self.tmp_dir = tmp_dir

    def _new(
        self,
        pcollection,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
    ):
        if shape is None:
            shape = self.shape
        if chunks is None:
            chunks = self.chunks
        if dtype is None:
            dtype = self.dtype
        if partition_row_counts is None:
            partition_row_counts = self.partition_row_counts
        return ndarray_pcollection(
            self.pipeline,
            pcollection,
            shape,
            chunks,
            dtype,
            partition_row_counts,
            self.tmp_dir,
        )

    def _new_or_copy(
        self,
        pcollection,
        shape=None,
        chunks=None,
        dtype=None,
        partition_row_counts=None,
        copy=True,
    ):
        if copy:
            return self._new(pcollection, shape, chunks, dtype, partition_row_counts)
        else:
            self.pcollection = pcollection
            if shape is not None:
                self.shape = shape
            if chunks is not None:
                self.chunks = chunks
            if dtype is not None:
                self.dtype = dtype
            if partition_row_counts is not None:
                self.partition_row_counts = partition_row_counts
        return self

    def close(self):
        shutil.rmtree(self.tmp_dir)

    # methods to convert to/from regular ndarray - mainly for testing
    @classmethod
    def from_ndarray(cls, pipeline, arr, chunks):
        shape = arr.shape
        ci = get_chunk_indices(shape, chunks)
        chunk_indices = pipeline | beam.Create(ci)

        # use the first component of chunk index as an index (assumes rows are one chunk wide)
        pcollection = chunk_indices | beam.Map(
            lambda chunk_index: (
                chunk_index[0],
                _read_chunk_from_arr(arr, chunks, chunk_index),
            )
        )
        return cls(pipeline, pcollection, shape, chunks, arr.dtype)

    @classmethod
    def from_zarr(cls, pipeline, zarr_file):
        """
        Read a Zarr file as an ndarray_pcollection object.
        """
        z = zarr.open(zarr_file, mode="r")
        shape, chunks = z.shape, z.chunks
        ci = get_chunk_indices(shape, chunks)
        chunk_indices = pipeline | beam.Create(ci)

        # use the first component of chunk index as an index (assumes rows are one chunk wide)
        pcollection = chunk_indices | beam.Map(
            lambda chunk_index: (
                chunk_index[0],
                read_zarr_chunk(zarr.open(zarr_file, mode="r"), chunks, chunk_index),
            )
        )
        return cls(pipeline, pcollection, shape, chunks, z.dtype)

    def asndarray(self):
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

        pcollection_row_counts = [len(arr) for arr in local_rows]
        assert pcollection_row_counts == list(self.partition_row_counts), (
            "PCollection row counts: %s; partition row counts: %s"
            % (pcollection_row_counts, self.partition_row_counts)
        )
        arr = np.concatenate(local_rows)
        assert arr.shape[0] == builtins.sum(self.partition_row_counts), (
            "PCollection #rows: %s; partition row counts total: %s"
            % (arr.shape[0], builtins.sum(self.partition_row_counts))
        )
        return arr

    def _write_zarr(self, store, chunks, write_chunk_fn):
        partitioned_pcollection = (
            self.pcollection
        )  # TODO: repartition if needed (currently Spark-only)
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)
        partitioned_pcollection | gensym("write_zarr") >> beam.Map(write_chunk_fn)

        result = self.pipeline.run()
        result.wait_until_finish()

    def to_zarr(self, zarr_file, chunks):
        """
        Write an anndata object to a Zarr file.
        """
        self._write_zarr(zarr_file, chunks, _write_chunk_zarr(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an anndata object to a Zarr file on GCS.
        """
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        self._write_zarr(
            store, chunks, _write_chunk_zarr_gcs(gcs_path, gcs_project, gcs_token)
        )

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
                new_pcollection, new_shape, new_shape, partition_row_counts=new_shape
            )
        elif axis == 1:  # sum of each row
            new_pcollection = a.pcollection | gensym("sum") >> beam.Map(
                lambda pair: (pair[0], np.sum(pair[1], axis=1))
            )
            return a._new(new_pcollection, (a.shape[0],), (a.chunks[0],))
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
                new_pcollection, new_shape, new_shape, partition_row_counts=new_shape
            )
        return NotImplemented

    # TODO: more calculation methods here

    # TODO: for Beam we should be able to avoid materializing everything - defer all the computations (even shapes, row partitions, etc)!

    # Distributed ufunc internal implementation

    def _unary_ufunc(self, func, dtype=None, copy=True):
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1]))
        )
        return self._new_or_copy(new_pcollection, dtype=dtype, copy=copy)

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1], pair[1]))
        )
        return self._new_or_copy(new_pcollection, dtype=dtype, copy=copy)

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        other = asarray(other)  # materialize
        # TODO: should send 'other' as a Beam side input
        new_pcollection = self.pcollection | gensym(func.__name__) >> beam.Map(
            lambda pair: (pair[0], func(pair[1], other))
        )
        return self._new_or_copy(new_pcollection, dtype=dtype, copy=copy)

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
            return self._new_or_copy(new_pcollection, dtype=dtype, copy=copy)
        return NotImplemented

    # Slicing

    def _boolean_array_index_dist(self, item):
        subset = item
        # materialize index PCollection to ndarray
        if isinstance(subset, ndarray_pcollection):
            subset = subset.asndarray()
        partition_row_subsets = self._copartition(subset)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
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

        # leave new chunks undefined since they are not necessarily equal-sized
        return self._new(
            new_pcollection,
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
                new_pcollection,
                shape=new_shape,
                chunks=new_chunks,
                partition_row_counts=self.partition_row_counts,
            )
        subset = item[1]
        # materialize index PCollection to ndarray
        if isinstance(subset, ndarray_pcollection):
            subset = subset.asndarray()
        new_pcollection = self.pcollection | gensym("column_subset") >> beam.Map(
            lambda pair: (pair[0], pair[1][item])
        )
        new_num_cols = builtins.sum(subset)
        new_shape = (self.shape[0], new_num_cols)
        new_chunks = (self.chunks[0], new_num_cols)
        return self._new(
            new_pcollection,
            shape=new_shape,
            chunks=new_chunks,
            partition_row_counts=self.partition_row_counts,
        )

    def _row_subset(self, item):
        subset = item[0]
        # materialize index PCollection to ndarray
        if isinstance(subset, ndarray_pcollection):
            subset = subset.asndarray()
        partition_row_subsets = self._copartition(subset)
        new_partition_row_counts = [builtins.sum(s) for s in partition_row_subsets]
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

        # leave new chunks undefined since they are not necessarily equal-sized
        return self._new(
            new_pcollection,
            shape=new_shape,
            partition_row_counts=new_partition_row_counts,
        )

    def _copartition(self, arr):
        partition_row_subsets = np.split(
            arr, np.cumsum(self.partition_row_counts)[0:-1]
        )
        if len(partition_row_subsets[-1]) == 0:
            partition_row_subsets = partition_row_subsets[0:-1]
        return partition_row_subsets
