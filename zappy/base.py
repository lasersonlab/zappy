import builtins
import copy as cp
import numbers
import numpy as np
import sys

from functools import partial

from zappy.zarr_util import (
    get_chunk_indices,
    read_zarr_chunk,
    write_chunk,
    write_chunk_gcs,
)

from numpy import *  # include everything in base numpy

npd = sys.modules[__name__]


def asarray(a):
    if isinstance(a, ZappyArray):
        return a.asndarray()
    return np.asarray(a)


def _delegate_to_np(func):
    """Delegate to numpy if the first arg is not a ZappyArray"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ZappyArray):
            return func(*args, **kwargs)
        # delegate to the equivalent in numpy
        return getattr(np, func.__name__)(*args, **kwargs)

    return delegated_func


def _delegate_to_np_dist(func):
    """Delegate to numpy if the first arg is not a ZappyArray"""

    def delegated_func(*args, **kwargs):
        if len(args) > 0 and isinstance(args[0], ZappyArray):
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
def argmax(a, axis=None):
    return a.argmax(axis)


@_delegate_to_np
def amin(a, axis=None):
    return a.min(axis)


@_delegate_to_np
def argmin(a, axis=None):
    return a.argmin(axis)


@_delegate_to_np
def sum(a, axis=None):
    return a.sum(axis)


@_delegate_to_np
def prod(a, axis=None):
    return a.sum(axis)


@_delegate_to_np
def all(a, axis=None):
    return a.all(axis)


@_delegate_to_np
def any(a, axis=None):
    return a.any(axis)


@_delegate_to_np
def mean(a, axis=None):
    return a.mean(axis)


@_delegate_to_np
def median(a):
    # note this is not a distributed implementation
    return np.median(a.asndarray())


class ZappyArray:
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
        obj = out if out else cp.copy(self)
        for key, value in kwargs.items():
            if key != "out":
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
        # subclasses should implement this to repartition to equal-sized chunks (except the last partition, which may be smaller)
        return NotImplemented

    def _repartition_if_necessary(self, chunks):
        # if all except last partition have c rows...
        # ... then no need to shuffle, since already partitioned correctly
        if all([count == chunks[0] for count in self.partition_row_counts[:-1]]):
            return self
        else:
            return self._repartition_chunks(chunks)

    def to_zarr(self, zarr_file, chunks):
        """
        Write an ZappyArray object to a Zarr file.
        """
        repartitioned = self._repartition_if_necessary(chunks)
        repartitioned._write_zarr(zarr_file, chunks, write_chunk(zarr_file))

    def to_zarr_gcs(self, gcs_path, chunks, gcs_project, gcs_token="cloud"):
        """
        Write an ZappyArray object to a Zarr file on GCS.
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

    # Array conversion (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#array-methods)

    def astype(self, dtype, copy=True):
        out = None if copy else self
        dtype = dtype if isinstance(dtype, np.dtype) else np.dtype(dtype)
        return self._unary_ufunc(lambda x: x.astype(dtype), out=out, dtype=dtype)

    def copy(self):
        return self._new()

    # Calculation methods (https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.ndarray.html#calculation)

    def mean(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.mean, axis)
        return self._calc_mean(axis)

    def argmax(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.argmax, axis)
        return self._calc_func_axis_distributive(np.argmax, axis)

    def min(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.amin, axis)
        return self._calc_func_axis_distributive(np.amin, axis)

    def argmin(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.argmin, axis)
        return self._calc_func_axis_distributive(np.argmin, axis)

    def sum(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.sum, axis)
        return self._calc_func_axis_distributive(np.sum, axis)

    def prod(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.prod, axis)
        return self._calc_func_axis_distributive(np.prod, axis)

    def all(self, axis=None):
        if axis == 1:
            return self._calc_func_axis_rowwise(np.all, axis)
        return self._calc_func_axis_distributive(np.all, axis)

    def any(self, axis=None):
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
        return self.asndarray().__getitem__(item)

    def _boolean_array_index_dist(self, item):
        return NotImplemented

    def _column_subset(self, item):
        return NotImplemented

    def _row_subset(self, item):
        return NotImplemented

    # Utility methods
    # TODO: document

    def _copartition(self, arr, partition_row_counts):
        if arr.dtype == np.dtype(int):  # indexes
            cum = np.cumsum(partition_row_counts)[0:-1]
            index_breaks = np.searchsorted(arr, cum)
            splits = np.array(np.split(arr, index_breaks))
            if len(splits[-1]) == 0:
                splits = np.array(splits[0:-1])
            offsets = np.insert(cum, 0, 0)  # add a leading 0
            partition_row_subsets = splits - offsets
            return partition_row_subsets
        else:  # values
            return self._copartition_values(arr, partition_row_counts)

    def _copartition_values(self, arr, partition_row_counts):
        partition_row_subsets = np.split(arr, np.cumsum(partition_row_counts)[0:-1])
        if len(partition_row_subsets[-1]) == 0:
            partition_row_subsets = partition_row_subsets[0:-1]
        return partition_row_subsets

    def _partition_row_counts(self, partition_row_subsets):
        dtype = partition_row_subsets[0].dtype
        if dtype == np.dtype(bool):
            return [int(builtins.sum(s)) for s in partition_row_subsets]
        elif dtype == np.dtype(int):
            return [len(s) for s in partition_row_subsets]
        return NotImplemented

    def _materialize_index(self, index):
        if isinstance(index, slice):
            return index
        return asarray(index)

    def _compute_dim(self, dim, subset):
        all_indices = slice(None, None, None)
        if isinstance(subset, slice):
            if subset == all_indices:
                return dim
            else:
                return len(np.zeros((dim))[subset])
        elif subset.dtype == np.dtype(int):
            return len(subset)
        return builtins.sum(subset)

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
        return npd.add(self, other, out=self)

    def __isub__(self, other):
        return npd.subtract(self, other, out=self)

    def __imul__(self, other):
        return npd.multiply(self, other, out=self)

    def __idiv__(self, other):
        return npd.multiply(self, other, out=self)

    def __itruediv__(self, other):
        return npd.true_divide(self, other, out=self)

    def __ifloordiv__(self, other):
        return npd.floor_divide(self, other, out=self)

    def __imod__(self, other):
        return npd.mod(self, other, out=self)

    def __ipow__(self, other):
        return npd.power(self, other, out=self)

    def __ilshift__(self, other):
        return npd.lshift(self, other, out=self)

    def __irshift__(self, other):
        return npd.rshift(self, other, out=self)

    def __iand__(self, other):
        return npd.bitwise_and(self, other, out=self)

    def __ior__(self, other):
        return npd.bitwise_or(self, other, out=self)

    def __ixor__(self, other):
        return npd.bitwise_xor(self, other, out=self)

    # Matrix Multiplication

    # TODO: __matmul__
