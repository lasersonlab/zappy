import builtins
import numbers
import numpy as np
import sys

from functools import partial

from zap.zarr_spark import (
    get_chunk_indices,
    read_zarr_chunk,
    write_chunk,
    write_chunk_gcs,
)

from numpy import *  # include everything in base numpy

npd = sys.modules[__name__]


def asarray(a):
    if isinstance(a, ndarray_dist):
        return a.asndarray()
    return np.asarray(a)


def _delegate_to_np(func):
    """Delegate to numpy if the first arg is not an ndarray_dist"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_dist):
            return func(*args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)

    return delegated_func


def _delegate_to_np_dist(func):
    """Delegate to numpy if the first arg is not an ndarray_dist"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ndarray_dist):
            return args[0]._dist_ufunc(func, args[1:], **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)

    return delegated_func


# Implement numpy ufuncs
# see https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#available-ufuncs
UFUNC_NAMES = (
    # Math operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#math-operations)
    "add",
    "subtract",
    "multiply",
    "divide",
    "logaddexp",
    "logaddexp2",
    "true_divide",
    "floor_divide",
    "negative",
    "positive",
    "power",
    "remainder",
    "mod",
    "fmod",
    # 'divmod', # not implemented since returns pair
    "absolute",
    "abs",
    "fabs",
    "rint",
    "sign",
    "heaviside",
    "conj",
    "exp",
    "exp2",
    "log",
    "log2",
    "log10",
    "expm1",
    "log1p",
    "sqrt",
    "square",
    "cbrt",
    "reciprocal",
    # Trigonometric functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#trigonometric-functions)
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "deg2rad",
    "rad2deg",
    # Bit-twiddling functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#bit-twiddling-functions)
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "invert",
    "left_shift",
    "right_shift",
    # Comparison functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#comparison-functions)
    "greater",
    "greater_equal",
    "less",
    "less_equal",
    "not_equal",
    "equal",
    "logical_and",
    "logical_or",
    "logical_xor",
    "logical_not",
    "maximum",
    "minimum",
    "fmax",
    "fmin",
    # Floating functions (https://docs.scipy.org/doc/numpy-1.14.0/reference/ufuncs.html#floating-functions)
    "isfinite",
    "isinf",
    "isnan",
    "isnat",
    "fabs",
    "signbit",
    "copysign",
    "nextafter",
    "spacing",
    # 'modf', # not implemented since returns pair
    "ldexp",
    # 'frexp', # not implemented since returns pair
    "fmod",
    "floor",
    "ceil",
    "trunc",
)
for ufunc_name in UFUNC_NAMES:
    ufunc = getattr(np, ufunc_name)
    setattr(npd, ufunc_name, _delegate_to_np_dist(ufunc))

# Implementations of selected functions in the numpy package


@_delegate_to_np
def sum(a, axis=None):
    return a.sum(axis)


@_delegate_to_np
def mean(a, axis=None):
    return a.mean(axis)


@_delegate_to_np
def median(a):
    # note this is not a distributed implementation
    return np.median(a.asndarray())


class ndarray_dist:

    def __init__(self, shape, chunks, dtype, partition_row_counts=None):
        self.shape = shape
        self.chunks = chunks
        self.dtype = dtype
        if partition_row_counts is None:
            partition_row_counts = [chunks[0]] * (shape[0] // chunks[0])
            remaining = shape[0] % chunks[0]
            if remaining != 0:
                partition_row_counts.append(remaining)
        self.partition_row_counts = partition_row_counts

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

    def asndarray(self):
        inputs = self._compute()
        partition_row_counts = self.partition_row_counts
        local_row_counts = [len(arr) for arr in inputs]
        assert local_row_counts == list(partition_row_counts), (
            "Local row counts: %s; partition row counts: %s"
            % (local_row_counts, partition_row_counts)
        )
        arr = np.concatenate(inputs)
        assert arr.shape[0] == builtins.sum(partition_row_counts), (
            "Local #rows: %s; partition row counts total: %s"
            % (arr.shape[0], builtins.sum(partition_row_counts))
        )
        return arr

    def _compute(self):
        """
        :return: a list of array chunks
        """
        return NotImplemented

    def _repartition_chunks(self, chunks):
        # subclasses should implement this to repartition to equal-sized chunks
        return NotImplemented

    def _repartition_if_necessary(self, chunks):
        if all([count == chunks[0] for count in self.partition_row_counts[:-1]]):
            return self
        else:
            return self._repartition_chunks(chunks)

    def to_zarr(self, zarr_file, chunks):
        """
        Write an ndarray_dist object to a Zarr file.
        """
        repartitioned = self._repartition_if_necessary(chunks)
        repartitioned._write_zarr(zarr_file, chunks, write_chunk(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an ndarray_dist object to a Zarr file on GCS.
        """
        repartitioned = self._repartition_if_necessary(chunks)
        import gcsfs.mapping

        gcs = gcsfs.GCSFileSystem(gcs_project, token=gcs_token)
        store = gcsfs.mapping.GCSMap(gcs_path, gcs=gcs)
        repartitioned._write_zarr(
            store, chunks, write_chunk_gcs(gcs_path, gcs_project, gcs_token)
        )

    def _write_zarr(self, store, chunks, write_chunk_fn):
        return NotImplemented

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        return NotImplemented

    def sum(self, axis=None):
        return NotImplemented

    # TODO: more calculation methods here

    # Distributed ufunc internal implementation

    def _dist_ufunc(self, func, args, dtype=None, copy=True):
        # unary ufunc
        if len(args) == 0:
            return self._unary_ufunc(func, dtype, copy)
        # binary ufunc
        elif len(args) == 1:
            other = args[0]
            if self is other:
                return self._binary_ufunc_self(func, dtype, copy)
            elif isinstance(other, numbers.Number) or other.ndim == 1:
                return self._binary_ufunc_broadcast_single_row_or_value(
                    func, other, dtype, copy
                )
            elif self.shape[0] == other.shape[0] and other.shape[1] == 1:
                return self._binary_ufunc_broadcast_single_column(
                    func, other, dtype, copy
                )
            elif self.shape == other.shape:
                return self._binary_ufunc_same_shape(func, other, dtype, copy)
        else:
            print("_dist_ufunc %s not implemented for %s" % (func, args))
            return NotImplemented

    def _unary_ufunc(self, func, dtype=None, copy=True):
        return NotImplemented

    def _binary_ufunc_self(self, func, dtype=None, copy=True):
        return NotImplemented

    def _binary_ufunc_broadcast_single_row_or_value(
        self, func, other, dtype=None, copy=True
    ):
        return NotImplemented

    def _binary_ufunc_broadcast_single_column(self, func, other, dtype=None, copy=True):
        return NotImplemented

    def _binary_ufunc_same_shape(self, func, other, dtype=None, copy=True):
        return NotImplemented

    # Slicing implementation

    def __getitem__(self, item):
        all_indices = slice(None, None, None)
        if isinstance(item, numbers.Number):
            return self._integer_index(item)
        elif isinstance(item, np.ndarray) and item.dtype == bool:
            return self._boolean_array_index(item)
        elif isinstance(item, ndarray_dist) and item.dtype == bool:
            return self._boolean_array_index_dist(item)
        elif isinstance(item[0], slice) and item[0] == all_indices:
            return self._column_subset(item)
        elif isinstance(item[1], slice) and item[1] == all_indices:
            return self._row_subset(item)
        return NotImplemented

    def _integer_index(self, item):
        # TODO: not scalable for large arrays
        return self.asndarray().__getitem__(item)

    def _boolean_array_index(self, item):
        # TODO: not scalable for large arrays
        return self.asndarray().__getitem__(item)

    def _boolean_array_index_dist(self, item):
        return NotImplemented

    def _column_subset(self, item):
        return NotImplemented

    def _row_subset(self, item):
        return NotImplemented

    # Utility methods

    def _copartition(self, arr, partition_row_counts):
        partition_row_subsets = np.split(arr, np.cumsum(partition_row_counts)[0:-1])
        if len(partition_row_subsets[-1]) == 0:
            partition_row_subsets = partition_row_subsets[0:-1]
        return partition_row_subsets

    def _partition_row_counts(self, partition_row_subsets):
        return [int(builtins.sum(s)) for s in partition_row_subsets]

    # Arithmetic, matrix multiplication, and comparison operations (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#arithmetic-matrix-multiplication-and-comparison-operations)

    # Python operator overloading, all delegate to ufunc implementations in this package

    # Comparison operators

    def __lt__(self, other):
        return npd.less(self, other, dtype=bool)

    def __le__(self, other):
        return npd.less_equal(self, other, dtype=bool)

    def __gt__(self, other):
        return npd.greater(self, other, dtype=bool)

    def __ge__(self, other):
        return npd.greater_equal(self, other, dtype=bool)

    def __eq__(self, other):
        return npd.equal(self, other, dtype=bool)

    def __ne__(self, other):
        return npd.not_equal(self, other, dtype=bool)

    # Truth value of an array (bool)

    # TODO: __nonzero__

    # Unary operations

    def __neg__(self):
        return npd.negative(self)

    def __pos__(self):
        return npd.positive(self)

    def __abs__(self):
        return npd.abs(self)

    def __invert__(self):
        return npd.invert(self)

    # Arithmetic

    def __add__(self, other):
        return npd.add(self, other)

    def __sub__(self, other):
        return npd.subtract(self, other)

    def __mul__(self, other):
        return npd.multiply(self, other)

    def __div__(self, other):
        return npd.divide(self, other)

    def __truediv__(self, other):
        return npd.true_divide(self, other)

    def __floordiv__(self, other):
        return npd.floor_divide(self, other)

    def __mod__(self, other):
        return npd.mod(self, other)

    # TODO: not implemented since returns pair
    # def __divmod__(self, other):
    #     return npd.div_mod(self, other)

    def __pow__(self, other):
        return npd.power(self, other)

    def __lshift__(self, other):
        return npd.lshift(self, other)

    def __rshift__(self, other):
        return npd.rshift(self, other)

    def __and__(self, other):
        return npd.bitwise_and(self, other)

    def __or__(self, other):
        return npd.bitwise_or(self, other)

    def __xor__(self, other):
        return npd.bitwise_xor(self, other)

    # Arithmetic, in-place

    def __iadd__(self, other):
        return npd.add(self, other, copy=False)

    def __isub__(self, other):
        return npd.subtract(self, other, copy=False)

    def __imul__(self, other):
        return npd.multiply(self, other, copy=False)

    def __idiv__(self, other):
        return npd.multiply(self, other, copy=False)

    def __itruediv__(self, other):
        return npd.true_divide(self, other, copy=False)

    def __ifloordiv__(self, other):
        return npd.floor_divide(self, other, copy=False)

    def __imod__(self, other):
        return npd.mod(self, other, copy=False)

    def __ipow__(self, other):
        return npd.power(self, other, copy=False)

    def __ilshift__(self, other):
        return npd.lshift(self, other, copy=False)

    def __irshift__(self, other):
        return npd.rshift(self, other, copy=False)

    def __iand__(self, other):
        return npd.bitwise_and(self, other, copy=False)

    def __ior__(self, other):
        return npd.bitwise_or(self, other, copy=False)

    def __ixor__(self, other):
        return npd.bitwise_xor(self, other, copy=False)

    # Matrix Multiplication

    # TODO: __matmul__
