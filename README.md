# Automated annotation of natural languages using selected toolchains

![Version](https://img.shields.io/pypi/v/nlpannotator)
![License: MIT](https://img.shields.io/github/license/ssciwr/argumentation-management)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/ssciwr/argumentation-management/CI)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/argumentation-management)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_argumentation-management&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/argumentation-management)

*This project just had its first version release and is still under development.*

## Description

The `nlpannotator` package serves as modular toolchain to combine different natural language processing (nlp) tools to annotate texts (sentencizing, tokenization, part-of-speech (POS) and lemma).

## Options

All input options are provided in an input dictionary. Two pre-set toolchains can be used: `fast` using [spaCy](https://spacy.io/) for all annotations; `accurate` using [SoMaJo](https://github.com/tsproisl/SoMaJo) for sentencizing and tokenization, and [stanza](https://stanfordnlp.github.io/stanza/) for POS and lemma; and `manual` where any combination of spaCy, stanza, SoMaJo, [Flair](https://github.com/flairNLP/flair), [Treetagger](https://treetaggerwrapper.readthedocs.io/en/latest/) can be used, given the tool supports the selected annotation and language.

## Installation

Install the project and its dependencies from [PyPi](https://pypi.org/project/nlpannotator/1.0.0/):  
```
pip install nlpannotator
```
The language models need to be installed separately. You can make use of the convenience script [here](install_models.py) which installs all language models for all languages that have been implemented for spaCy and stanza.

## Usage

Take a look at the [DemoNotebook](./docs/demo-notebook.ipynb) or run it on [Binder]().
