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


## Languages


## Demo notebook

Take a look at the [DemoNotebook](./docs/demo-notebook.ipynb) or run it on [Binder](https://mybinder.org/v2/gh/ssciwr/argumentation-management/HEAD?labpath=.%2Fdocs%2Fdemo-notebook.ipynb).


## Questions and bug reports

Please ask questions / submit bug reports using our [issue tracker](https://github.com/ssciwr/argumentation-management/issues).

## Contribute

Contributions are wellcome. Please fork the nlpannotator repo and open a Pull Request for any changes to the code. These will be reviewed and merged by our team.
Make sure that your contributions are [clean](https://flake8.pycqa.org/en/latest/), [properly formatted](https://github.com/psf/black) and for any new modules [follow the general design principle](https://github.com/ssciwr/argumentation-management/blob/main/nlpannotator/mstanza.py).