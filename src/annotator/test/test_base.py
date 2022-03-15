import pytest
import json
import base as be
import tempfile


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
    "output": "test",
    "tool": "stanza",
    "cwb_dict": {
        "corpus_name": "test",
        "corpus_dir": "/home/jovyan/shared/corpora/",
        "registry_dir": "/home/jovyan/shared/registry/",
    },
    "stanza_dict": {"processors": "tokenize, pos, lemma"},
}


@pytest.fixture
def get_obj():
    obj = be.encode_corpus(be.prepare_run.get_encoding(test_dict))
    return obj


@pytest.mark.skip
def test_get_cores():
    pass


@pytest.mark.dictname("test/test_files/input")
def test_load_input_dict(init_dict):
    mydict = be.prepare_run.load_input_dict("input")
    assert mydict == init_dict


@pytest.mark.dictname("test/test_files/input2")
def test_validate_input_dict(init_dict):
    be.prepare_run.validate_input_dict(init_dict)


@pytest.mark.dictname("test/test_files/input_short")
def test_update_dict(init_dict):
    mydict = be.prepare_run.load_input_dict("input")
    mydict = mydict["spacy_dict"]
    mydict = be.prepare_run.update_dict(mydict)
    assert mydict == init_dict


@pytest.mark.dictname("test/test_files/input_stanza")
def test_activate_procs(init_dict):
    mydict = be.prepare_run.load_input_dict("input")
    mydict = mydict["stanza_dict"]
    mydict = be.prepare_run.update_dict(mydict)
    mydict_content = be.prepare_run.activate_procs(mydict, "stanza_")
    assert mydict_content == init_dict
    # test for empty procs
    mydict["processors"] = None
    with pytest.raises(ValueError):
        assert be.prepare_run.activate_procs(mydict, "stanza_")


# chunker class - to be completed
# def test_chunker


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


# out_object to be tested in spacy/stanza
# test the encode_corpus class
# everything except the actual cwb command
# we do not want to install it in CI/CD
# to use dockerfile for workflow is left for later


def test_encode_vrt(get_obj):
    obj = get_obj
    line = " "
    line = obj._get_s_attributes(line)
    test_line = " -S s "
    assert line == test_line
    line = obj._get_p_attributes(line)
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

    answers = iter(["n", "y", "{}".format(tmp.name), "test", "test", "y"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    assert obj.setup() is True
