from setuptools import setup, find_packages

import os

version_file = open(os.path.join("./annotator", "VERSION"))
version = version_file.read().strip()

requires_file = open(os.path.join("./annotator", "requirements.txt"))
requirements = [line.strip() for line in requires_file]

setup(
    name="annotator",
    version=version,
    description="annotator source files",
    author="Dr. Inga Ulusoy, Christian Delavier",
    license="MIT License",
    packages=find_packages(),
    install_requires=requirements,
)
