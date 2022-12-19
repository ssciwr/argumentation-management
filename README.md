# Automated annotation of natural languages using selected toolchains

![Version](https://img.shields.io/pypi/v/nlpannotator)
![License: MIT](https://img.shields.io/github/license/ssciwr/argumentation-management)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ssciwr/argumentation-management/ci.yml?branch=main)
![codecov](https://img.shields.io/codecov/c/github/ssciwr/argumentation-management)
![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=ssciwr_argumentation-management&metric=alert_status)
![Language](https://img.shields.io/github/languages/top/ssciwr/argumentation-management)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![OpenSSF Best Practices](https://bestpractices.coreinfrastructure.org/projects/6698/badge)](https://bestpractices.coreinfrastructure.org/projects/6698)

*This project just had its first version release and is still under development.*

## Description

The `nlpannotator` package serves as modular toolchain to combine different natural language processing (nlp) tools to annotate texts (sentencizing, tokenization, part-of-speech (POS) and lemma).

Tools that can be combined are:
- [spaCy](https://spacy.io/) (sentencize, tokenize, POS, lemma)
- [stanza](https://stanfordnlp.github.io/stanza/) (sentencize, tokenize, POS, lemma)
- [SoMaJo](https://github.com/tsproisl/SoMaJo) (sentencize, tokenize)
- [Flair](https://github.com/flairNLP/flair) (POS)
- [Treetagger](https://treetaggerwrapper.readthedocs.io/en/latest/) (tokenize, POS, lemma)
These tools can be combined in any desired fashion, to target either maximum efficiency or accuracy.

## Installation

Install the project and its dependencies from [PyPi](https://pypi.org/project/nlpannotator/1.0.0/):  
```
pip install nlpannotator
```
The language models need to be installed separately. You can make use of the convenience script [here](https://github.com/ssciwr/argumentation-management/blob/main/install_latest_models.sh) which installs all language models for all languages that have been implemented for spaCy and stanza.

## Options

All input options are provided in an input dictionary. Two pre-set toolchains can be used: `fast` using [spaCy](https://spacy.io/) for all annotations; `accurate` using [SoMaJo](https://github.com/tsproisl/SoMaJo) for sentencizing and tokenization, and [stanza](https://stanfordnlp.github.io/stanza/) for POS and lemma; and `manual` where any combination of spaCy, stanza, SoMaJo, [Flair](https://github.com/flairNLP/flair), [Treetagger](https://treetaggerwrapper.readthedocs.io/en/latest/) can be used, given the tool supports the selected annotation and language.

| Keyword | Default setting | Possible options | Description |
| ------- | --------------- | ---------------- | ----------- |
| `input` | `example_en.txt`  | | Name of the text file containing the raw text for annotation |
| `corpus_name` | `test` | | Name of the corpus that is generated |
| `language` | `en` | [see below](#Languages) | Language of the text to annotate |
| `processing_option` | `manual` | `fast, accurate, manual` | Select the tool pipeline - `fast` and `accurate` provide you with good default options for English |
| `processing_type`| `sentencize, tokenize, pos, lemma` | [see below](#Processors) |
| `tool`  | `spacy, spacy, spacy, spacy` | [see below](#Tools) | Tool to use for each of the four annotation types |
| `output_format` | `xml` | `xml, vrt` | Format of the generated annotated text file |
| `encoding` | `yes` | `yes, no` | Directly encode the annotated text file into [cwb](https://cwb.sourceforge.io/) |

## Tools
The available annotation tools are listed below, and can be set using the following keywords:
- [spaCy](https://spacy.io/): `spacy`
- [stanza](https://stanfordnlp.github.io/stanza/): `stanza`
- [SoMaJo](https://github.com/tsproisl/SoMaJo): `somajo`
- [Flair](https://github.com/flairNLP/flair): `flair`
- [Treetagger](https://treetaggerwrapper.readthedocs.io/en/latest/): `treetagger`

## Processors
The available processors depend on the selected tool. This is a summary of the possible options:
| Tool | Available processors |
| ---- | -------------------- |
| `spacy` | `sentencize, tokenize, pos, lemma` |
| `stanza` | `sentencize, tokenize, pos, lemma` |
| `somajo` | `sentencize, tokenize` |
| `flair` | `pos` |
| `treetagger` | `tokenize, pos, lemma` |
Some of the processors depend on each other. For example, `pos` and `lemma` are only possible after `sentencize` and `tokenize`. `tokenize` depends on `sentencize`. 

## Languages
The availabe languages depend on the selected tool. So far, the following languages have been added to the pipeline (there may be additional language models available for the respective tool, but they have not been added to this package - for stanza, the pipeline will still run and load the model on demand).
| Tool | Available languages |
| ---- | -------------------- |
| `spacy` | `en, de, fr, it, ja, pt, ru, es` |
| `stanza` | load on demand from [available stanza models](https://stanfordnlp.github.io/stanza/available_models.html) |
| `somajo` | `en, de` |
| `flair` | `en, de` |
| `treetagger` | `en, de, fr, es` (both tokenization and pos/lemma) |
| `treetagger` | `bg, nl, et, fi, gl, it, kr, la, mn, pl, ru, sk, sw` (only pos/lemma) |

## Input/Output
`nlpannotator` expects a raw text file as an input, together with an input dictionary that specifies the selected options. The input dictionary is also printed out when a run is initiated, so that the selected options are stored and can be looked up at a later time.
Both of these can be provided through a [Jupyter](https://jupyter.org/) interface as in the [Demo Notebook](#Demo).

The output that is generated is either of `vrt` format (for cwb) or `xml`. Both output formats can directly be encoded into cwb.

## Demo notebook

Take a look at the [DemoNotebook](./docs/demo-notebook.ipynb) or run it on [Binder](https://mybinder.org/v2/gh/ssciwr/argumentation-management/HEAD?labpath=.%2Fdocs%2Fdemo-notebook.ipynb).


## Questions and bug reports

Please ask questions / submit bug reports using our [issue tracker](https://github.com/ssciwr/argumentation-management/issues).

## Contribute

Contributions are wellcome. Please fork the nlpannotator repo and open a Pull Request for any changes to the code. These will be reviewed and merged by our team.
Make sure that your contributions are [clean](https://flake8.pycqa.org/en/latest/), [properly formatted](https://github.com/psf/black) and for any new modules [follow the general design principle](https://github.com/ssciwr/argumentation-management/blob/main/nlpannotator/mstanza.py).

Take a look at the [source code documentation](file:///home/iulusoy/projects/argumentation-project/argumentation-management/docs/build/html/modules.html).

The additions must have at least have 80% test coverage.

## Releases

A summary of the releases and release notes are available [here](https://github.com/ssciwr/argumentation-management/releases).