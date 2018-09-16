# Zap - distributed processing with Numpy and Zarr

Zap is for distributed processing of chunked Numpy arrays on engines like Apache Spark and Apache Beam.

[![Build Status](https://travis-ci.org/lasersonlab/zap.svg?branch=master)](https://travis-ci.org/lasersonlab/zap)

The `zap.base` module defines a `ndarray_dist` class that exposes the same interface as `numpy.ndarray`, and which
is backed by distributed storage and processing. The array is broken into chunks, which is typically loaded from Zarr,
and each chunk is processed independently.

There are a few engines provided:
* **direct** - for eager in-memory processing
* **spark** - for processing using Spark
* **beam** - for processing using Beam or Google Dataflow
* **executor** - for processing using Python's [concurrent.futures.Executor]

Beam currently only runs on Python 2.

Hopefully the executor engine will make it possible to run on [Pywren] at some point.

Full coverage of the `numpy.ndarray` interface is _not_ provided. Only enough has been implemented to support running
parts of [Scanpy], as demonstrated in the [Single Cell Experiments] repo.

## Testing

There is a test suite for all the engines, covering both Python 2 and 3.

Create and activate a Python 3 virtualenv, and install the requirements:

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Or for Python 2:

```
virtualenv venv2
. venv2/bin/activate
pip install -r requirements.txt
pip install -e .
```

Run Tests:

```
pytest
```

Or run everything in one go with tox:

```
pip install tox
tox
```

Coverage:

```
pip install pytest-cov
pytest --cov-report html --cov=zap
open htmlcov/index.html
```

[Scanpy]: https://scanpy.readthedocs.io/
[Single Cell Experiments]: https://github.com/lasersonlab/single-cell-experiments
[concurrent.futures.Executor]: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor
[Pywren]: http://pywren.io/