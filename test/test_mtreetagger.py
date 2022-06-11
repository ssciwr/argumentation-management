import re
import pytest
from .context import base as be
from .context import mtreetagger as mtt


@pytest.fixture
def read_data_en():
    return be.prepare_run.get_text("test/test_files/example_en.txt")


@pytest.fixture
def read_data_de():
    return be.prepare_run.get_text("test/test_files/example_de.txt")


@pytest.fixture
def read_test_en():
    with open("test/test_files/example_en_treetagger.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def read_test_de():
    with open("test/test_files/example_de_treetagger.vrt", "r") as f:
        data = f.read()
    return data


def test_tokenize(read_data_en, read_data_de, read_test_de, read_test_en):

    text_de = read_data_de
    data_de = read_test_de

    assert mtt.tokenize(text_de, "de")[0] == data_de

    text_en = read_data_en
    data_en = read_test_en

    assert mtt.tokenize(text_en, "en")[0] == data_en
