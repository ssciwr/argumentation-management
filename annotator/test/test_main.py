import pytest
import main as mn
import base as be


@pytest.fixture
def load_dict():
    mydict = be.PrepareRun.load_input_dict("./test/test_files/input")
    return mydict


@pytest.fixture
def data_en():
    return "This is a sentence."


@pytest.fixture
def test_en():
    return ["This is a sentence."]


@pytest.fixture
def test_en_somajo():
    return ["This is a sentence ."]


def test_call_spacy(load_dict, data_en, test_en):
    load_dict["processing_type"] = "sentencize"
    out_obj = mn.call_spacy(load_dict, data_en)
    assert out_obj.sentences == test_en


def test_call_stanza(load_dict, data_en, test_en):
    load_dict["stanza_dict"]["processors"] = "tokenize,pos"
    out_obj = mn.call_stanza(load_dict, data_en)
    assert out_obj.sentences == test_en
    load_dict["stanza_dict"]["processors"] = "tokenize"
    out_obj = mn.call_stanza(load_dict, data_en)
    assert out_obj.sentences == test_en


def test_call_somajo(load_dict, data_en, test_en_somajo):
    load_dict["somajo_dict"]["processors"] = "sentencize"
    load_dict["somajo_dict"]["model"] = "en_PTB"
    out_obj = mn.call_somajo(load_dict, data_en)
    assert out_obj.sentences == test_en_somajo


def test_call_treetagger(load_dict, data_en):
    load_dict["treetagger_dict"]["processors"] = "pos"
    load_dict["treetagger_dict"]["lang"] = "en"
    out_obj = mn.call_treetagger(load_dict, data_en)
    assert out_obj.ptags == []


def test_call_flair(load_dict, data_en):
    load_dict["flair_dict"]["processors"] = "pos"
    load_dict["flair_dict"]["model"] = "pos"
    out_obj = mn.call_flair(load_dict, data_en)
    assert out_obj.ptags == []
