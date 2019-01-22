import builtins
import copy as cp
import numbers
import numpy as np
import zarr

from functools import partial

from zappy.zarr_util import (
    get_chunk_indices,
    read_zarr_chunk,
    write_chunk,
    write_chunk_gcs,
    write_n_chunk_copies,
    write_n_chunk_copies_gcs,
)


class ZappyArray(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, shape, chunks, dtype, partition_row_counts=None):
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        if partition_row_counts is None:
            partition_row_counts = [chunks[0]] * (shape[0] // chunks[0])
            remaining = shape[0] % chunks[0]
            if remaining != 0:
                partition_row_counts.append(remaining)
        self.partition_row_counts = partition_row_counts

    def _new(self, **kwargs):
        """Copy or update this object with the given keyword parameters."""
        out = kwargs.get("out")
        if isinstance(out, tuple) and len(out) > 0:
            out = out[0]  # TODO: handle multiple values
        obj = out if out else cp.copy(self)
        for key, value in kwargs.items():
            if key != "out" and not (key == "dtype" and value is None):
                setattr(obj, key, value)
        return obj

    @property
    def ndim(self):
        return len(self.shape)

    # Load and store methods

    @classmethod
    def from_ndarray(cls, sc, arr, chunks):
        return NotImplemented

    @classmethod
    def from_zarr(cls, sc, zarr_file):
        return NotImplemented

    @staticmethod
    def _read_chunks(arr, chunks):
        shape = arr.shape
        func = partial(read_zarr_chunk, arr, chunks)
        chunk_indices = get_chunk_indices(shape, chunks)
        return func, chunk_indices

    @staticmethod
    def _array_chunks_to_ndarray(array_chunks, partition_row_counts):
        local_row_counts = [len(arr) for arr in array_chunks]
        assert local_row_counts == list(partition_row_counts), (
            "Local row counts: %s; partition row counts: %s"
            % (local_row_counts, partition_row_counts)
        )
        arr = np.concatenate(array_chunks)
        assert arr.shape[0] == builtins.sum(partition_row_counts), (
            "Local #rows: %s; partition row counts total: %s"
            % (arr.shape[0], builtins.sum(partition_row_counts))
        )
        return arr

    def asndarray(self):
        return ZappyArray._array_chunks_to_ndarray(
            self._compute(), self.partition_row_counts
        )

    def __array__(self, dtype=None, **kwargs):
        # respond to np.asarray
        x = self.asndarray()
        if dtype and x.dtype != dtype:
            x = x.astype(dtype)
        return x

    def _compute(self):
        """
        :return: a list of array chunks
        """
        return NotImplemented

    def _repartition_chunks(self, chunks):
        # subclasses should implement this to repartition to equal-sized chunks (except the last partition, which may be smaller)
        return NotImplemented

    def _repartition_if_necessary(self, chunks):
        # if all except last partition have c rows...
        # ... then no need to shuffle, since already partitioned correctly
        if all([count == chunks[0] for count in self.partition_row_counts[:-1]]):
            return self
        else:
            return self._repartition_chunks(chunks)

    def to_zarr(self, zarr_file, chunks, ncopies=1):
        """
        Write an ZappyArray object to a Zarr file.
        """
        if ncopies != 1:
            assert self.shape[0] % chunks[0] == 0
            shape = (self.shape[0] * ncopies, self.shape[1])
            zarr.open(zarr_file, mode="w", shape=shape, chunks=chunks, dtype=self.dtype)
            self._write_zarr(
                zarr_file,
                chunks,
                write_n_chunk_copies(zarr_file, self.shape[0], ncopies),
            )
            return
        zarr.open(
            zarr_file, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype
        )
        repartitioned = self._repartition_if_necessary(chunks)
        repartitioned._write_zarr(zarr_file, chunks, write_chunk(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud", ncopies=1):
        """
        Write an ZappyArray object to a Zarr file on GCS.
        """
        if ncopies != 1:
            assert self.shape[0] % chunks[0] == 0
            shape = (self.shape[0] * ncopies, self.shape[1])
            import gcsfs.mapping

            gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
            store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
            zarr.open(store, mode="w", shape=shape, chunks=chunks, dtype=self.dtype)
            self._write_zarr(
                store,
                chunks,
                write_n_chunk_copies_gcs(
                    gcs_path, gcs_project, gcs_token, self.shape[0], ncopies
                ),
            )
            return

        repartitioned = self._repartition_if_necessary(chunks)
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        zarr.open(store, mode="w", shape=self.shape, chunks=chunks, dtype=self.dtype)
        repartitioned._write_zarr(
            store, chunks, write_chunk_gcs(gcs_path, gcs_project, gcs_token)
        )

    def _write_zarr(self, store, chunks, write_chunk_fn, ncopies=1):
        return NotImplemented

    # Array conversion (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#array-methods)

    def astype(self, dtype, copy=True):
        out = None if copy else self
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        return self._unary_ufunc(lambda x: x.astype(dtype), out=out, dtype=dtype)

    def copy(self):
        return self._new()

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.mean, axis)
        return self._calc_mean(axis)

    def argmax(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.argmax, axis)
        return self._calc_func_axis_distributive(np.argmax, axis)

    def min(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.amin, axis)
        return self._calc_func_axis_distributive(np.amin, axis)

    def argmin(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.argmin, axis)
        return self._calc_func_axis_distributive(np.argmin, axis)

    def sum(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.sum, axis)
        return self._calc_func_axis_distributive(np.sum, axis)

    def prod(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.prod, axis)
        return self._calc_func_axis_distributive(np.prod, axis)

    def all(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.all, axis)
        return self._calc_func_axis_distributive(np.all, axis)

    def any(self, axis, out=None, dtype=None, **kwargs):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.any, axis)
        return self._calc_func_axis_distributive(np.any, axis)

    # TODO: more calculation methods here
    # Not distributive: ptp, cumsum, var, std, cumprod
    # Don't take an axis: clip, conj, round
    # Other: trace (two axes!)

    def _calc_mean(self, axis=None):
        return NotImplemented

    def _calc_func_axis_rowwise(self, func, axis):
        # Calculation method that takes an axis argument and axis == 1
        return NotImplemented

    def _calc_func_axis_distributive(self, func, axis):
        # Calculation method that takes an axis argument, and is distributive.
        # Distributive in this context means that the result can be computed in pieces
        # and combined using the function. So f(a, b, c, d) = f(f(a, b), f(c, d))
        return NotImplemented

    # Distributed ufunc internal implementation

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":
            # TODO: handle dtype generically
            if ufunc.__name__ in (
                "greater",
                "greater_equal",
                "less",
                "less_equal",
                "not_equal",
                "equal",
            ):
                kwargs["dtype"] = bool
            if len(inputs) > 0 and isinstance(inputs[0], ZappyArray):
                return inputs[0]._dist_ufunc(ufunc, inputs[1:], **kwargs)
        return NotImplemented

    def _dist_ufunc(self, func, args, out=None, dtype=None):
        # unary ufunc
        if len(args) == 0:
            return self._unary_ufunc(func, out=out, dtype=dtype)
        # binary ufunc
        elif len(args) == 1:
            other = args[0]
            if self is other:
                return self._binary_ufunc_self(func, out=out, dtype=dtype)
            elif isinstance(other, numbers.Number) or other.ndim == 1:
                return self._binary_ufunc_broadcast_single_row_or_value(
                    func, other, out=out, dtype=dtype
                )
            elif self.shape[0] == other.shape[0] and other.shape[1] == 1:
                return self._binary_ufunc_broadcast_single_column(
                    func, other, out=out, dtype=dtype
                )
            elif self.shape == other.shape:
                return self._binary_ufunc_same_shape(func, other, out=out, dtype=dtype)
        else:
            print("_dist_ufunc %s not implemented for %s" % (func, args))
            return NotImplemented

    def _unary_ufunc(self, func, out=None, dtype=None):
        return NotImplemented

    def _binary_ufunc_self(self, func, out=None, dtype=None):
        return NotImplemented

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, out=None, dtype=None
    ):
        return NotImplemented

    def _binary_ufunc_broadcast_single_column(self, func, other, out=None, dtype=None):
        return NotImplemented

    def _binary_ufunc_same_shape(self, func, other, out=None, dtype=None):
        return NotImplemented

    # Slicing implementation

    def __getitem__(self, item):
        all_indices = slice(None, None, None)
        if isinstance(item, numbers.Number):
            return self._integer_index(item)
        elif isinstance(item, (np.ndarray, ZappyArray)) and item.dtype == bool:
            return self._boolean_array_index_dist(item)
        elif isinstance(item[0], slice) and item[0] == all_indices:
            return self._column_subset(item)
        elif isinstance(item[1], slice) and item[1] == all_indices:
            return self._row_subset(item)
        return NotImplemented

    def _integer_index(self, item):
        # TODO: not scalable for large arrays
        return np.asarray(self).__getitem__(item)

    def _boolean_array_index_dist(self, item):
        return NotImplemented

    def _column_subset(self, item):
        return NotImplemented

    def _row_subset(self, item):
        return NotImplemented

    # Utility methods
    # TODO: document

    @staticmethod
    def _copartition(arr, partition_row_counts, shrink=False):
        """Partition an array or slice according to the given row counts.
        The array can be of indexes, or values.
        """
        if isinstance(arr, slice):  # slice
            cum = np.cumsum(partition_row_counts)[0:-1]
            offsets = np.insert(cum, 0, 0)  # add a leading 0
            start = arr.start if arr.start is not None else 0
            stop = arr.stop if arr.stop is not None else np.sum(partition_row_counts)
            # find the partition index where the slice starts and stops
            starti = np.searchsorted(cum, start, side="right")
            stopi = np.searchsorted(cum, stop)
            partition_row_subsets = [slice(0, 0)] * starti
            for i in range(starti, stopi + 1):
                subset_start = start - offsets[i] if i == starti else None
                subset_stop = stop - offsets[i] if i == stopi else None
                partition_row_subsets.append(slice(subset_start, subset_stop))
            if not shrink:
                partition_row_subsets.extend(
                    [slice(0, 0)] * (len(partition_row_counts) - (stopi + 1))
                )
            return partition_row_subsets
        elif arr.dtype == np.dtype(int):  # indexes
            cum = np.cumsum(partition_row_counts)[0:-1]
            offsets = np.insert(cum, 0, 0)  # add a leading 0
            index_breaks = np.searchsorted(arr, cum)
            splits = np.split(arr, index_breaks)
            partition_row_subsets = []
            for index, split in enumerate(splits):
                if len(split) == 0:
                    partition_row_subsets.append(split)
                else:
                    partition_row_subsets.append(split - offsets[index])
            return partition_row_subsets
        else:  # values
            return ZappyArray._copartition_values(arr, partition_row_counts)

    @staticmethod
    def _copartition_values(arr, partition_row_counts):
        return np.split(arr, np.cumsum(partition_row_counts)[0:-1])

    @staticmethod
    def _partition_row_counts(partition_row_subsets, partition_row_counts=None):
        if isinstance(partition_row_subsets[0], slice):
            counts = []
            for i, subset in enumerate(partition_row_subsets):
                start = subset.start if subset.start is not None else 0
                stop = (
                    subset.stop if subset.stop is not None else partition_row_counts[i]
                )
                counts.append(stop - start)
            return counts
        dtype = partition_row_subsets[0].dtype
        if dtype == np.dtype(bool):
            return [int(builtins.sum(s)) for s in partition_row_subsets]
        elif dtype == np.dtype(int):
            return [len(s) for s in partition_row_subsets]
        return NotImplemented

    @staticmethod
    def _materialize_index(index):
        """Materialize index as an ndarray, or leave as a slice."""
        if isinstance(index, slice):
            return index
        return np.asarray(index)

    @staticmethod
    def _compute_dim(dim, subset):
        """Compute the dimension of an index or slice."""
        all_indices = slice(None, None, None)
        if isinstance(subset, slice):
            if subset == all_indices:
                return dim
            else:
                return len(np.zeros((dim))[subset])
        elif subset.dtype == np.dtype(int):
            return len(subset)
        return builtins.sum(subset)
