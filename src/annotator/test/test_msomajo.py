import pytest
import msomajo as mso
import base as be


@pytest.fixture
def read_data_en():
    return be.prepare_run.get_text("test/test_files/example_en.txt")


@pytest.fixture
def read_data_de():
    return be.prepare_run.get_text("test/test_files/example_de.txt")


@pytest.fixture
def read_test_de():
    with open("test/test_files/example_de_somajo.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def read_test_en():
    with open("test/test_files/example_en_somajo.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def load_dict():
    mydict = be.prepare_run.load_input_dict("./test/test_files/input")
    mydict["somajo_dict"]["model"] = "en_PTB"
    mydict["somajo_dict"]["processors"] = "sentencize", "tokenize"
    return mydict


def test_init(load_dict):
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    assert tokenized.jobs == ("sentencize", "tokenize")
    assert tokenized.model == "en_PTB"
    assert tokenized.sentencize
    assert tokenized.camelcase


def test_apply_to(read_data_en, read_data_de, load_dict):
    text_en = read_data_en
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    tokenized.apply_to(text_en)
    assert tokenized.doc[0][8].text == "software"
    text_de = read_data_de
    load_dict["somajo_dict"]["model"] = "de_CMC"
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    tokenized.apply_to(text_de)
    assert tokenized.doc[2][5].text == "dass"
