# Zappy - distributed processing with NumPy and Zarr

Zappy is for distributed processing of chunked NumPy arrays on engines like [Pywren], Apache Spark, and Apache Beam.

[![Build Status](https://travis-ci.org/lasersonlab/zappy.svg?branch=master)](https://travis-ci.org/lasersonlab/zappy)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Coverage Status](https://coveralls.io/repos/github/lasersonlab/zappy/badge.svg)](https://coveralls.io/github/lasersonlab/zappy)
[![PyPI version shields.io](https://img.shields.io/pypi/v/zappy.svg)](https://pypi.python.org/pypi/zappy/)

The `zappy.base` module defines a `ZappyArray` class that exposes the same interface as `numpy.ndarray`, and which
is backed by distributed storage and processing. The array is broken into chunks, and is typically loaded from [Zarr],
and each chunk is processed independently.

There are a few engines provided:
* **direct** - for eager in-memory processing
* **spark** - for processing using Spark
* **beam** - for processing using Beam or Google Dataflow
* **executor** - for processing using Python's [concurrent.futures.Executor], of which [Pywren] is a notable implementation

Beam currently only runs on Python 2.

Full coverage of the `numpy.ndarray` interface is _not_ provided. Only enough has been implemented to support running
parts of [Scanpy], as demonstrated in the [Single Cell Experiments] repo.

## Installation

```
pip install zappy
```

Alternatively, zappy can be installed using [Conda](https://conda.io/docs/) (most easily obtained via the [Miniconda Python distribution](https://conda.io/miniconda.html)):

```
conda install -c conda-forge zappy
```

## Demo

Take a look at the rendered [demo Jupyter notebook](demo.ipynb), or try it out yourself as follows.

Create and activate a Python 3 virtualenv, and install the requirements:

```
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install s3fs jupyter
```

Then run the notebook with:

```
jupyter notebook demo.ipynb
```

## Testing

There is a test suite for all the engines, covering both Python 2 and 3.

Run everything in one go with tox:

```
pip install tox
tox
```

Formatting:

```
pip install black
black zappy tests/* *.py
```

Coverage:

```
pip install pytest-cov
pytest --cov-report html --cov=zappy
open htmlcov/index.html
```

## Publishing

```
pip install twine
python setup.py sdist
twine upload -r pypi dist/zappy-0.1.0.tar.gz
```

If successful, the package will be available on [PyPI].

[Scanpy]: https://scanpy.readthedocs.io/
[Single Cell Experiments]: https://github.com/lasersonlab/single-cell-experiments
[concurrent.futures.Executor]: https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Executor
[PyPI]: https://pypi.org/project/zappy/
[Pywren]: http://pywren.io/
[Zarr]: https://zarr.readthedocs.io/
