import pytest
import base as be
import pipe as pe
import mstanza as ma
from tempfile import TemporaryDirectory


@pytest.fixture()
def load_data():
    data = be.prepare_run.get_text("./test/test_files/example_de.txt")
    return data


def test_integration_mstanza(load_data):

    # create temporary directories for the corpora
    out = TemporaryDirectory()

    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["input"] = "./test/test_files/example_de.txt"
    mydict["tool"] = "stanza, stanza, stanza, stanza"
    mydict["language"] = "de"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "manual"
    mydict["processing_type"] = "tokenize,pos,mwt,lemma"
    mydict["advanced_options"]["output_dir"] = "./test/out/"
    # validate the input dict
    be.prepare_run.validate_input_dict(mydict)
    # load the pipe object for updating dict with settings
    obj = pe.SetConfig(mydict)
    stanza_dict = obj.mydict["stanza_dict"]
    stanza_dict["dir"] = "./test/models/"
    data = be.prepare_run.get_text(mydict["input"])
    # initialize the pipeline with the dict
    stanza_pipe = ma.MyStanza(stanza_dict)
    # apply pipeline to data and encode
    stanza_pipe.apply_to(data).pass_results(mydict)
    # maybe here assert that written vrt is same as safe version
