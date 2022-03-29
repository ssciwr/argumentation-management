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
    assert obj.tool == ["spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]


def test_pipe_accurate(get_mydict):
    get_mydict["processing_option"] = "accurate"
    get_mydict["processing_type"] = " lemma, tokenize, pos  "
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "stanza", "stanza"]
    assert obj.processors == ["tokenize", "pos", "lemma"]


def test_pipe_manual(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "spacy, stanza, spacy"
    get_mydict["processing_type"] = "tokenize, pos, lemma"
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "stanza", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]


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