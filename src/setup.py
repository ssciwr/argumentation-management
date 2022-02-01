from setuptools import setup, find_packages

import os

version_file = open(os.path.join("./annotator", "VERSION"))
version = version_file.read().strip()

setup(
    name="Annotator",
    Version=version,
    description="Annotator source files",
    authors="Dr. Inga Ulusoy, Christian Delavier",
    license="MIT License",
    packages=find_packages(),
    install_requires=["numpy"],
)
