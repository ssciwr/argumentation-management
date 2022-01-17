import pytest
import json
import base as be


class get_sample:
    def __init__(self) -> None:
        self.text_dict = {
            "en": "./iued_test_original.txt",
            "de": "./iued_test_i_en53_pt-export.txt",
        }

    def get_sample(self, lang):
        with open(self.text_dict[lang], "r") as myfile:
            data_en = myfile.read().replace("\n", "")
        return data_en

    @staticmethod
    def set_input_dict(name):
        # load the default input dict
        with open("{}.json".format(name)) as f:
            mydict = json.load(f)
        return mydict


@pytest.mark.skip
def test_get_cores():
    pass


def test_load_input_dict():
    name = "input"
    mydict = be.prepare_run.load_input_dict(name)
    test_mydict = get_sample.set_input_dict("test/" + name)
    assert mydict == test_mydict


def test_update_dict():
    name = "input"
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
