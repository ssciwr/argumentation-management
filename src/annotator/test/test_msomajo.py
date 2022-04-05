import pytest
import annotator.msomajo as msm
import annotator.base as be


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


def test_tokenize(read_data_en, read_data_de, read_test_de, read_test_en):

    text_de = read_data_de
    data_de = read_test_de

    assert msm.tokenize(text_de, "de_CMC")[0] == data_de

    text_en = read_data_en
    data_en = read_test_en

    assert msm.tokenize(text_en, "en_PTB")[0] == data_en
