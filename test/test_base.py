import pytest
import unittest.mock
import json
import os
from .context import base as be
import tempfile


@pytest.fixture()
def get_path():
    return os.getcwd()


@pytest.fixture()
def init_dict(request):
    """Load the specified input dict."""
    marker = request.node.get_closest_marker("dictname")
    if marker == "None":
        # missing marker
        print("Missing a marker for reading the input dictionary.")
    else:
        name = marker.args[0]
    with open("{}.json".format(name)) as f:
        mydict = json.load(f)
    return mydict


test_dict = {
    "input": "input.txt",
    "tool": ["stanza", "stanza", "stanza"],
    "corpus_name": "test",
    "language": "en",
    "document_type": "text",
    "processing_option": "manual",
    "processing_type": ["tokenize", "pos", "lemma"],
    "advanced_options": {
        "output_dir": "./out/",
        "output_format": "STR",
        "corpus_dir": "./corpora/",
        "registry_dir": "./registry/",
    },
    "stanza_dict": {"processors": ["tokenize", "pos", "lemma"]},
}


@pytest.fixture
def get_obj():
    # obj = be.encode_corpus(be.prepare_run.get_encoding(test_dict))
    obj = be.encode_corpus(test_dict)
    return obj


@pytest.fixture
def get_obj_dec():
    obj = be.decode_corpus(test_dict)
    return obj


@pytest.mark.skip
def test_get_cores():
    pass


@pytest.mark.dictname("./test/test_files/input")
def test_load_input_dict(init_dict, get_path):
    mydict = be.prepare_run.load_input_dict("./annotator/input")
    assert mydict == init_dict


@pytest.mark.dictname("./test/test_files/input2")
def test_validate_input_dict(init_dict):
    be.prepare_run.validate_input_dict(init_dict)


def test_chunker():
    text = '<textid="1"> This is an example text. <subtextid="1"> It has some subtext. </subtext> </text> <textid="2"> Here is some more text. </text>'

    formated_text = text.replace(" ", "\n")

    tmp = tempfile.NamedTemporaryFile()

    tmp.write(formated_text.encode())
    tmp.seek(0)
    # print(tmp.read().decode())
    data = be.chunk_sample_text("{}".format(tmp.name))
    # print(data)
    # don't need this anymore
    tmp.close()

    check = [
        ['<textid="1"> ', "This is an example text. ", ""],
        ['<subtextid="1"> ', "It has some subtext. ", "</subtext> "],
        ["", "", "</text> "],
        ['<textid="2"> ', "Here is some more text. ", "</text>"],
    ]

    assert data == check


# OutObject to be tested in spacy/stanza
# test the encode_corpus class
# everything except the actual cwb command
# we do not want to install it in CI/CD
# to use dockerfile for workflow is left for later


def test_purge():

    inputs = [" ", "  "]
    outputs = ["", ""]

    for input, output in zip(inputs, outputs):
        assert be.OutObject.purge(input) == output


def test_encode_vrt(get_obj):

    obj = get_obj
    line = " "
    line = obj._get_s_attributes(line, stags=["s"])
    test_line = " -S s "
    assert line == test_line
    line = obj._get_p_attributes(line, ptags=["pos", "lemma"])
    test_line += "-P pos -P lemma "
    assert line == test_line


def test_setup(monkeypatch, get_obj):

    tmp = tempfile.TemporaryDirectory()
    obj = get_obj
    obj.encodedir = tmp.name

    monkeypatch.setattr("builtins.input", lambda _: "y")

    assert obj.setup() is True

    answers = iter(["n", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert obj.setup() is False

    answers = iter(["n", "y", "n", "{}".format(tmp.name), "test", "test", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert obj.setup() is True

    answers = iter(["n", "y", "y", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert obj.setup() is True

    answers = iter(["n", "y", "y", "n", "n"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert obj.setup() is False


@unittest.mock.patch("os.system")
def test_decode(os_system, get_path, get_obj_dec):

    path = get_path
    obj = get_obj_dec
    obj.decode_to_file()

    os_system.assert_called_with(
        "cd {} && cwb-decode -r {} {} -ALL > {}.out && cd {}".format(
            obj.corpusdir,
            obj.regdir,
            obj.corpusname,
            path + "/" + obj.corpusname,
            path + "/",
        )
    )

    obj.decode_to_file(verbose=False)

    os_system.assert_called_with(
        "cd {} && cwb-decode -C -r {} {} -ALL > {}.out && cd {}".format(
            obj.corpusdir,
            obj.regdir,
            obj.corpusname,
            path + "/" + obj.corpusname,
            path + "/",
        )
    )

    obj.decode_to_file(
        specific={
            "P-Attributes": ["test_p_0", "test_p_1"],
            "S-Attributes": ["test_s_0", "test_s_1"],
        }
    )

    os_system.assert_called_with(
        "cd {} && cwb-decode -r {} {} -P test_p_0 -P test_p_1 -S test_s_0 -S test_s_1 > {}.out && cd {}".format(
            obj.corpusdir,
            obj.regdir,
            obj.corpusname,
            path + "/" + obj.corpusname,
            path + "/",
        )
    )
