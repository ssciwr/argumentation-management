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
    corp = TemporaryDirectory()
    reg = TemporaryDirectory()
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict = pe.SetConfig.set_processors(mydict)
    mydict["tool"] = "stanza"
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)
    mydict["advanced_options"]["corpus_dir"] = "{}".format(corp.name)
    mydict["advanced_options"]["registry_dir"] = "{}".format(reg.name)

    # validate the input dict
    be.prepare_run.validate_input_dict(mydict)
    stanza_dict = mydict["stanza_dict"]
    stanza_dict["lang"] = "de"
    stanza_dict["processors"] = "tokenize,pos,mwt,lemma"
    stanza_dict["dir"] = "./test/models/"
    # stanza_dict = be.prepare_run.update_dict(stanza_dict)
    stanza_dict = be.prepare_run.activate_procs(stanza_dict, "stanza_")
    data = load_data
    # initialize the pipeline with the dict
    stanza_pipe = ma.MyStanza(stanza_dict)
    # apply pipeline to data and encode
    stanza_pipe.apply_to(data).pass_results(mydict)
    # maybe here assert that written vrt is same as safe version
