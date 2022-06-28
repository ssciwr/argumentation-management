from webbrowser import get
import pytest
import base as be
import pipe as pe

procstring = "tokenize, pos, lemma"


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
    assert obj.tool == ["stanza", "stanza", "stanza"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == []
    assert get_mydict["stanza_dict"]["processors"] == "tokenize,pos,lemma"


def test_pipe_manual_multiple(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "spacy, stanza, spacy"
    get_mydict["processing_type"] = procstring
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "stanza", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == ["tok2vec", "lemmatizer"]
    assert get_mydict["stanza_dict"]["processors"] == "pos"


def test_pipe_manual_one(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "spacy"
    get_mydict["processing_type"] = procstring
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["spacy", "spacy", "spacy"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["spacy_dict"]["processors"] == ["tok2vec", "tagger", "lemmatizer"]


def test_get_processors(get_mydict):
    get_mydict["processors"] = "lemma, tokenize, pos"
    obj = pe.SetConfig(get_mydict)
    processors = obj._get_processors(get_mydict["processors"])
    assert processors == ["lemma", "tokenize", "pos"]


def test_order_processors(get_mydict):
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
    assert obj.tool == ["stanza", "stanza", "stanza"]


def test_set_model_spacy(get_mydict):
    get_mydict["model"] = "mymodel"
    obj = pe.SetConfig(get_mydict)
    assert get_mydict["spacy_dict"]["model"] == "mymodel"
    get_mydict.pop("model", None)
    obj._set_model_spacy()
    assert get_mydict["spacy_dict"]["model"] == "en_core_web_md"
    get_mydict["language"] = "de"
    obj._set_model_spacy()
    assert get_mydict["spacy_dict"]["model"] == "de_core_news_md"


def test_set_model_stanza(get_mydict):
    get_mydict["model"] = "mymodel"
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "stanza"
    obj = pe.SetConfig(get_mydict)
    assert get_mydict["stanza_dict"]["model"] == "mymodel"
    get_mydict.pop("model", None)
    obj._set_model_stanza()
    assert get_mydict["stanza_dict"]["model"] is None


def test_set_model_somajo(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "somajo, somajo"
    get_mydict["processing_type"] = "sentencize, tokenize"
    obj = pe.SetConfig(get_mydict)
    assert obj.mydict["somajo_dict"]["model"] == "en_PTB"
    obj.mydict["language"] = "de"
    obj._set_model_somajo()
    assert obj.mydict["somajo_dict"]["model"] == "de_CMC"


def test_set_model_treetagger(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "treetagger, treetagger"
    get_mydict["processing_type"] = "tokenize, pos"
    obj = pe.SetConfig(get_mydict)
    assert obj.mydict["treetagger_dict"]["lang"] == "en"
    obj.mydict["language"] = "de"
    obj._set_model_treetagger()
    assert obj.mydict["treetagger_dict"]["lang"] == "de"


def test_set_model_flair(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "flair"
    get_mydict["processing_type"] = "pos"
    obj = pe.SetConfig(get_mydict)
    assert obj.mydict["flair_dict"]["model"] == "pos"
    get_mydict["tool"] = "flair, flair"
    get_mydict["processing_type"] = "pos, ner"
    obj = pe.SetConfig(get_mydict)
    assert obj.mydict["flair_dict"]["model"] == ["pos", "ner"]
    get_mydict["tool"] = "flair"
    get_mydict["processing_type"] = "pos"
    obj.mydict["language"] = "de"
    obj = pe.SetConfig(get_mydict)
    assert obj.mydict["flair_dict"]["model"] == "de-pos"
    obj.mydict["language"] = "fr"
    with pytest.raises(ValueError):
        obj._set_model_flair()


def test_set_processors(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "stanza"
    get_mydict["processing_type"] = procstring
    obj = pe.SetConfig(get_mydict)
    assert obj.tool == ["stanza", "stanza", "stanza"]
    assert obj.processors == ["tokenize", "pos", "lemma"]
    assert get_mydict["processing_type"] == ["tokenize", "pos", "lemma"]


def set_tool(get_mydict):
    get_mydict["processing_option"] = "manual"
    get_mydict["tool"] = "stanza"
    get_mydict["processing_type"] = procstring
    pe.SetConfig(get_mydict)
    assert get_mydict["tool"] == ["stanza", "stanza", "stanza"]
