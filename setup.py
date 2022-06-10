from setuptools import setup, find_packages

version_file = open("VERSION")
version = version_file.read().strip()

requires_file = open("requirements.txt")
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
