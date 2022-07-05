import pytest
import unittest.mock
import json
import os
import annotator.base as be
import annotator.mtreetagger as mtt
import annotator.mspacy as msp
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
    with open("{}".format(name)) as f:
        mydict = json.load(f)
    return mydict


@pytest.fixture
def data_en():
    return "This is a sentence."


@pytest.fixture
def load_dict():
    mydict = be.PrepareRun.load_input_dict("./test/data/input.json")
    mydict["treetagger_dict"]["lang"] = "en"
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    print(mydict)
    return mydict["treetagger_dict"], mydict["spacy_dict"]


@pytest.fixture
def get_doc(load_dict, data_en):
    annotated = mtt.MyTreetagger(load_dict[0])
    annotated.apply_to(data_en)
    return annotated.doc, annotated.jobs


@pytest.fixture
def test_token_en():
    token_en = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    token_en_annotated = [
        "<s>",
        "This\tDT\tthis\n",
        "is\tVBZ\tbe\n",
        "a\tDT\ta\n",
        "sentence\tNN\tsentence\n",
        ".\tSENT\t.\n",
        "</s>",
    ]
    return token_en, token_en_annotated


@pytest.fixture
def test_en():
    data = [
        "<s>",
        "This\tthis\n",
        "is\tbe\n",
        "a\ta\n",
        "sentence\tsentence\n",
        ".\t.\n",
        "</s>",
    ]
    return data


@pytest.fixture
def test_en_sentence():
    sentence = [
        ("This\tthis", 1),
        ("is\tbe", 2),
        ("a\ta", 3),
        ("sentence\tsentence", 4),
        (".\t.", 5),
    ]
    return sentence


@pytest.fixture
def test_en_sentence2():
    sentence = ["<s>\n", "This\n", "is\n", "a\n", "sentence\n", ".\n", "</s>\n"]
    return sentence


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
    obj = be.encode_corpus(test_dict)
    return obj


@pytest.fixture
def get_obj_dec():
    obj = be.decode_corpus(test_dict)
    return obj


@pytest.mark.dictname("./test/data/input.json")
def test_load_input_dict(init_dict):
    mydict = be.PrepareRun.load_input_dict("./data/input.json")
    assert mydict == init_dict


@pytest.mark.dictname("./test/data/input2.json")
def test_validate_input_dict(init_dict):
    be.PrepareRun.validate_input_dict(init_dict)


def test_iterate(load_dict, data_en, test_en_sentence2):
    annotated = msp.MySpacy(load_dict[1])
    annotated.apply_to(data_en)
    out_obj = msp.OutSpacy(annotated.doc, annotated.jobs, 0)
    out = []
    for sent in annotated.doc.sents:
        out.append("<s>\n")
        out = out_obj.iterate(out, sent)
        out.append("</s>\n")
    assert out == test_en_sentence2


def test_iterate_tokens(get_doc, test_token_en):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    token_list = out_obj.token_list(out_obj.doc)
    out = out_obj.iterate_tokens(test_token_en[0], token_list)
    assert out == test_token_en[1]


def test_token_list(get_doc):
    mylist = ["a", "n", "d"]
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    token_list = out_obj.token_list(mylist)
    assert token_list == mylist


def test_out_shortlist(get_doc, test_en, test_en_sentence):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    shortlist = out_obj.out_shortlist(test_en)
    assert shortlist == test_en_sentence


def test_compare_tokens(get_doc):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    mytoken1 = "English"
    mytoken2 = "is"
    assert out_obj._compare_tokens(mytoken1, mytoken1)
    assert not out_obj._compare_tokens(mytoken1, mytoken2)


def test_write_vrt():
    mystring = "abcdefgh"
    myfile = "test/out/test"
    be.OutObject.write_vrt(myfile, mystring)
    test_string = be.PrepareRun.get_text(myfile + ".vrt")
    assert test_string == mystring


def test_write_xml():
    mystring = "abcdefgh"
    myfile = "test/out/test"
    be.OutObject.write_xml("test", myfile, mystring)
    mystring2 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><corpus name="test"><text>"""
    mystring2 += mystring
    mystring2 += """</text></corpus>"""
    test_string = be.PrepareRun.get_text(myfile + ".xml")
    assert test_string == mystring2


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
    my_attr = "builtins.input"
    monkeypatch.setattr(my_attr, lambda _: "y")
    assert obj.setup() is True
    answers = iter(["n", "n"])
    monkeypatch.setattr(my_attr, lambda _: next(answers))
    assert obj.setup() is False
    answers = iter(["n", "y", "n", "{}".format(tmp.name), "test", "test", "y"])
    monkeypatch.setattr(my_attr, lambda _: next(answers))
    assert obj.setup() is True
    answers = iter(["n", "y", "y", "y"])
    monkeypatch.setattr(my_attr, lambda _: next(answers))
    assert obj.setup() is True
    answers = iter(["n", "y", "y", "n", "n"])
    monkeypatch.setattr(my_attr, lambda _: next(answers))
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
