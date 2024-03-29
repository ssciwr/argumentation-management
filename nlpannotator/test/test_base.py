import pytest
import unittest.mock
import json
import os
from pathlib import Path
import nlpannotator.base as be
import nlpannotator.mtreetagger as mtt
import nlpannotator.mspacy as msp
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


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
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    mydict["treetagger_dict"]["lang"] = "en"
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    print(mydict)
    return mydict["treetagger_dict"], mydict["spacy_dict"]


@pytest.mark.treetagger
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


@pytest.fixture
def test_dict(tmp_path):
    mydict = {
        "input": "input.txt",
        "tool": ["stanza", "stanza", "stanza"],
        "corpus_name": "test",
        "language": "en",
        "document_type": "text",
        "processing_option": "manual",
        "processing_type": ["tokenize", "pos", "lemma"],
        "advanced_options": {
            "output_dir": Path(tmp_path / "out").as_posix(),
            "output_format": "STR",
            "corpus_dir": Path(tmp_path / "corpora").as_posix(),
            "registry_dir": Path(tmp_path / "registry").as_posix(),
        },
        "stanza_dict": {"processors": ["tokenize", "pos", "lemma"]},
    }
    return mydict


@pytest.fixture
def get_obj_enc(test_dict):
    obj = be.EncodeCorpus(test_dict)
    return obj


@pytest.fixture
def get_obj_dec(test_dict):
    obj = be.DecodeCorpus(test_dict)
    return obj


@pytest.mark.dictname(pkg / "data" / "input.json")
def test_load_input_dict(init_dict):
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    assert mydict == init_dict


@pytest.mark.dictname(pkg / "data" / "input2.json")
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


@pytest.mark.treetagger
def test_iterate_tokens(get_doc, test_token_en):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    token_list = out_obj.token_list(out_obj.doc)
    out = out_obj.iterate_tokens(test_token_en[0], token_list)
    assert out == test_token_en[1]


@pytest.mark.treetagger
def test_token_list(get_doc):
    mylist = ["a", "n", "d"]
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    token_list = out_obj.token_list(mylist)
    assert token_list == mylist


@pytest.mark.treetagger
def test_out_shortlist(get_doc, test_en, test_en_sentence):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    shortlist = out_obj.out_shortlist(test_en)
    assert shortlist == test_en_sentence


@pytest.mark.treetagger
def test_compare_tokens(get_doc):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    mytoken1 = "English"
    mytoken2 = "is"
    assert out_obj._compare_tokens(mytoken1, mytoken1)
    assert not out_obj._compare_tokens(mytoken1, mytoken2)


def test_get_names():
    attrdict = be.OutObject.get_names()
    assert attrdict["stanza_names"]["proc_sent"] == "tokenize"
    assert attrdict["spacy_names"]["proc_lemma"] == "lemmatizer"


def test_purge():
    inputs = [" ", "  "]
    outputs = ["", ""]
    for input, output in zip(inputs, outputs):
        assert be.OutObject.purge(input) == output


def test_write_vrt(tmp_path):
    mystring = "abcdefgh"
    myfile = tmp_path / "test"
    be.OutObject.write_vrt(myfile.as_posix(), [mystring])
    test_string = be.PrepareRun.get_text(myfile.as_posix() + ".vrt")
    assert test_string == mystring


def test_write_xml(tmp_path):
    mystring = "abcdefgh"
    myfile = tmp_path / "test"
    be.OutObject.write_xml("test", myfile.as_posix(), [mystring])
    mystring2 = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><corpus name="test"><text>"""
    mystring2 += mystring
    mystring2 += """</text></corpus>"""
    test_string = be.PrepareRun.get_text(myfile.as_posix() + ".xml")
    assert test_string == mystring2


def test_fix_path():
    path = "/home/jovyan/"
    assert be.EncodeCorpus.fix_path(path) == path
    path = "/home/jovyan"
    assert be.EncodeCorpus.fix_path(path) == path + "/"


def test_encode_corpus(get_obj_enc, test_dict):
    assert get_obj_enc.tool == test_dict["tool"]
    assert get_obj_enc.jobs == test_dict["processing_type"]
    path = test_dict["advanced_options"]["corpus_dir"]
    path = be.EncodeCorpus.fix_path(path)
    assert get_obj_enc.corpusdir == path
    assert get_obj_enc.encodedir == path
    path = test_dict["advanced_options"]["registry_dir"]
    path = be.EncodeCorpus.fix_path(path)
    assert get_obj_enc.regdir == path
    assert get_obj_enc.corpusname == test_dict["corpus_name"]
    outname = test_dict["advanced_options"]["output_dir"] + test_dict["corpus_name"]
    assert get_obj_enc.outname == outname


def test_get_s_attributes(get_obj_enc):
    stags = ["s"]
    line = ""
    line = get_obj_enc._get_s_attributes(line, stags)
    assert line == "-S s "
    line = "something "
    line = get_obj_enc._get_s_attributes(line, stags)
    assert line == "something -S s "


def test_get_p_attributes(get_obj_enc):
    ptags = ["lemma"]
    line = ""
    line = get_obj_enc._get_p_attributes(line, ptags)
    assert line == "-P lemma "
    line = "something "
    line = get_obj_enc._get_p_attributes(line, ptags)
    assert line == "something -P lemma "


def test_setup(get_obj_enc):
    assert get_obj_enc.setup()


def test_encode_vrt(get_obj_enc):
    obj = get_obj_enc
    line = " "
    line = obj._get_s_attributes(line, stags=["s"])
    test_line = " -S s "
    assert line == test_line
    line = obj._get_p_attributes(line, ptags=["pos", "lemma"])
    test_line += "-P pos -P lemma "
    assert line == test_line


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
