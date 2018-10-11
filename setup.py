#!/usr/bin/env python

from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="zappy",
    version="0.1.0",
    description="Distributed processing with NumPy and Zarr.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    author="Tom White",
    author_email="tom.e.white@gmail.com",
    url="https://github.com/lasersonlab/zappy",
)
