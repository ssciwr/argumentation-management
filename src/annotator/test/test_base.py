import pytest
import json
import spacy as sp
import base as be

class get_sample:
    def __init__(self) -> None:
        self.text_dict = {
            "en": "./iued_test_original.txt",
            "de": "./iued_test_i_en53_pt-export.txt",
        }

    @staticmethod
    def set_input_dict(name):
        # load the default input dict
        with open("{}.json".format(name)) as f:
            mydict = json.load(f)
        return mydict


text = """<s>
This	PRON	this
is	AUX	be
an	DET	a
example	NOUN	example
.	PUNCT	.
</s>
<s>
And	CCONJ	and
here	ADV	here
we	PRON	we
go	VERB	go
.	PUNCT	.
</s>"""


@pytest.mark.skip
def test_get_cores():
    pass


def test_load_input_dict():
    name = "input"
    mydict = be.prepare_run.load_input_dict(name)
    test_mydict = get_sample.set_input_dict("test/" + name)
    assert mydict == test_mydict


def test_update_dict():
    name = "test/input"
    mydict = be.prepare_run.load_input_dict(name)
    mydict = mydict["spacy_dict"]
    mydict = be.prepare_run.update_dict(mydict)
    test_mydict = get_sample.set_input_dict("test/input_short")
    assert mydict == test_mydict


def test_activate_procs():
    name = "input"
    mydict = be.prepare_run.load_input_dict(name)
    mydict = mydict["stanza_dict"]
    mydict = be.prepare_run.update_dict(mydict)
    mydict = be.prepare_run.activate_procs(mydict, "stanza_")
    test_mydict = get_sample.set_input_dict("test/input_stanza")
    assert mydict == test_mydict


# chunker class - to be completed
# def test_chunker
# out_object to be tested in spacy/stanza
# test the encode_corpus class
# everything except the actual cwb command
# we do not want to install it in CI/CD
# to use dockerfile for workflow is left for later
def test_encode_vrt():
    obj = be.encode_corpus("test", "test", ["tokenize", "pos", "lemma"], "stanza")
    line = " "
    line = obj._get_s_attributes(line)
    test_line = " -S s "
    assert line == test_line
    line = obj._get_p_attributes(line)
    test_line += "-P pos -P lemma "
    assert line == test_line
