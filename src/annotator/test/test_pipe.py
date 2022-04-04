from webbrowser import get
import pytest
import pipe as pe
import base as be


@pytest.fixture()
def get_mydict():
    mydict = be.prepare_run.load_input_dict("./test/test_files/input")
    return mydict


def test_pipe_fast(get_mydict):
    get_mydict["processing_option"] = "fast"
    get_mydict["processing_type"] = "lemma, tokenize, pos"
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "spacy", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]


def test_pipe_accurate(get_mydict):
    get_mydict["processing_option"] = "accurate"
    get_mydict["processing_type"] = " lemma, tokenize, pos  "
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "stanza", "stanza"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == ["tok2vec"]
    assert get_mydict["stanza_dict"]["processors"] == "pos,lemma"


def test_pipe_manual_multiple(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "spacy, stanza, spacy"
    get_mydict["processing_type"] = "tokenize, pos, lemma"
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "stanza", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == ["tok2vec", "lemmatizer"]
    assert get_mydict["stanza_dict"]["processors"] == "pos"


def test_pipe_manual_one(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "spacy"
    get_mydict["processing_type"] = "tokenize, pos, lemma"
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "spacy", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == ["tok2vec", "tagger", "lemmatizer"]


def test_get_processors(get_mydict):
    get_mydict["processors"] = "lemma, tokenize, pos"
    obj = pe.SetConfig(get_mydict)
    processors = obj._get_processors(get_mydict["processors"])
    assert processors == ["lemma", "tokenize", "pos"]


def test_order_proessors(get_mydict):
    get_mydict["processors"] = " lemma, tokenize, pos  "
    obj = pe.SetConfig(get_mydict)
    processors = obj._get_processors(get_mydict["processors"])
    obj._order_processors(processors)
    assert obj.processors == ["tokenize", "pos", "lemma"]


def test_get_tools(get_mydict):
    get_mydict["processors"] = " lemma, tokenize, pos  "
    obj = pe.SetConfig(get_mydict)
    processors = obj._get_processors(get_mydict["processors"])
    obj._order_processors(processors)
    obj._get_tools()
    assert obj.tool == ["spacy", "stanza", "stanza"]


def test_set_model_spacy(get_mydict):
    get_mydict["model"] = "mymodel"
    obj = pe.SetConfig(get_mydict)
    assert obj.model == "mymodel"
    get_mydict.pop("model", None)
    get_mydict["document_type"] = "text"
    obj = pe.SetConfig(get_mydict)
    assert obj.model == "en_core_web_md"
    get_mydict["document_type"] = "scientific"
    obj = pe.SetConfig(get_mydict)
    assert obj.model == "en_core_sci_md"
    get_mydict["document_type"] = "historic"
    obj = pe.SetConfig(get_mydict)
    assert obj.model == "en_core_web_md"
    get_mydict["language"] = "de"


def test_set_model_stanza():
    pass


def test_set_processors(get_mydict):
    obj = pe.SetConfig(get_mydict)
