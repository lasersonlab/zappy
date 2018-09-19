import itertools
import math
import zarr

try:
    from itertools import accumulate
except ImportError:
    # see https://docs.python.org/dev/library/itertools.html#itertools.accumulate
    import operator

    def accumulate(iterable, func=operator.add):
        "Return running totals"
        # accumulate([1,2,3,4,5]) --> 1 3 6 10 15
        # accumulate([1,2,3,4,5], operator.mul) --> 1 2 6 24 120
        it = iter(iterable)
        try:
            total = next(it)
        except StopIteration:
            return
        yield total
        for element in it:
            total = func(total, element)
            yield total


# Utility functions for reading and writing a Zarr array chunk; these are designed to be run as Spark tasks.
# Assume that the row lengths are small enough that the entire row fits into a Zarr chunk; in
# other words, the chunk width is the same as the row width. Also each task reads/writes a single chunk.
#
# Possible matrix operations:
# * Add or remove columns. Adjust chunk width. Easy to handle since row partitioning does not change.
# * Add or remove rows. Changes row partitioning. Simplest way to handle is to shuffle with the chunk as the key.
#   See repartition_chunks. May
#   be able to be more sophisticated with a clever Spark coalescer that can read from other partitions.
# * Matrix multiplication. Multiplying by a matrix on the right preserves partitioning, so only chunk width needs to
#   change.


def get_chunk_indices(shape, chunks):
    """
    Return all the indices (coordinates) for the chunks in a zarr array, even empty ones.
    """
    return [
        (i, j)
        for i in range(int(math.ceil(float(shape[0]) / chunks[0])))
        for j in range(int(math.ceil(float(shape[1]) / chunks[1])))
    ]


def get_chunk_sizes(shape, chunks):
    def sizes(length, chunk_length):
        res = [chunk_length] * (length // chunk_length)
        if length % chunk_length != 0:
            res.append(length % chunk_length)
        return res

    return itertools.product(sizes(shape[0], chunks[0]), sizes(shape[1], chunks[1]))


def read_zarr_chunk(arr, chunks, chunk_index):
    return arr[
        chunks[0] * chunk_index[0] : chunks[0] * (chunk_index[0] + 1),
        chunks[1] * chunk_index[1] : chunks[1] * (chunk_index[1] + 1),
    ]


def read_chunk(file):
    """
    Return a function to read a chunk by coordinates from the given file.
    """

    def read_one_chunk(chunk_index):
        """
        Read a zarr chunk specified by coordinates chunk_index=(a,b).
        """
        z = zarr.open(file, mode="r")
        return read_zarr_chunk(z, z.chunks, chunk_index)

    return read_one_chunk


def write_chunk(file):
    """
    Return a function to write a chunk by index to the given file.
    """

    def write_one_chunk(index_arr):
        """
        Write a partition index and numpy array to a zarr store. The array must be the size of a chunk, and not
        overlap other chunks.
        """
        index, arr = index_arr
        z = zarr.open(file, mode="r+")
        chunk_size = z.chunks
        z[chunk_size[0] * index : chunk_size[0] * (index + 1), :] = arr

    return write_one_chunk


def write_chunk_gcs(gcs_path, gcs_project, gcs_token):
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


def calculate_partition_boundaries(chunks, partition_row_counts):
    # Generate a list of offsets, so that k[i] is the number of rows before the i-th partition
    # Then turn this into a row range for each partition
    k = list(accumulate([0] + partition_row_counts))
    partition_row_ranges = list(zip(k, k[1:]))
    total_rows = k[-1]
    new_num_partitions = ((total_rows - 1) // chunks[0]) + 1
    return partition_row_ranges, total_rows, new_num_partitions


def extract_partial_chunks(iterator, chunks):
    """
    For a given partition, we now know the start and end row numbers, so use that along with the new chunk size
    to break the rows into new (partial) chunks that are labelled with the new index number. Partial chunks will
    be shuffled using the new index number as key to bring together all the partial chunks for a given new index
    number.
    """
    # iterator is a single entry of ((row_start, row_end), array), where row_end is exclusive

    c = chunks[0]  # the chunk size for rows
    key, val = list(iterator)
    k_i, k_i_next = key
    tuples = []
    for x in range(k_i - k_i % c, k_i_next, c):  # iterate over overlapping chunks
        start, end = max(k_i, x), min(k_i_next, x + c)
        start_offset, end_offset = start - k_i, end - k_i
        partial_chunk = val[start_offset:end_offset]
        new_index = start // c
        new_start_offset, new_end_offset = (start - new_index * c, end - new_index * c)
        tuples.append((new_index, ((new_start_offset, new_end_offset), partial_chunk)))
    return tuples
