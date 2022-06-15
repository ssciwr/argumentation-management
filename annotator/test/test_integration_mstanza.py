import pytest

# from .context import base as be
import base as be

# from .context import pipe as pe
import pipe as pe

# from .context import mstanza as ma
import mstanza as ma
from tempfile import TemporaryDirectory


@pytest.fixture()
def load_data():
    data = be.prepare_run.get_text("./test/test_files/example_de.txt")
    return data


def test_integration_mstanza(load_data):

    # read in input.json
    mydict = be.prepare_run.load_input_dict("input")
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
    # apply pipeline to data
    stanza_pipe.apply_to(data)
    # we should not need start ..?
    start = 0
    out_obj = ma.OutStanza(stanza_pipe.doc, stanza_pipe.jobs, start=start, islist=False)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.get_ptags()
    stags = out_obj.get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    # add = False
    # if not add:
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
    # elif add:
    # encode_obj = be.encode_corpus(mydict)
    # encode_obj.add_tags_to_corpus(mydict, ptags, stags)
    # maybe here assert that written vrt is same as safe version
