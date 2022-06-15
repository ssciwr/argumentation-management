import pytest

# from .context import base as be
import base as be

# from .context import msomajo as mso
import msomajo as mso

# from .context import mtreetagger as mtt
import mtreetagger as mtt
from tempfile import TemporaryDirectory


@pytest.fixture
def setup():
    mydict = be.prepare_run.load_input_dict("input")
    mydict["input"] = "./test/test_files/example_en.txt"
    be.prepare_run.validate_input_dict(mydict)
    text = be.prepare_run.get_text(mydict["input"])

    return mydict, text


def test_integration_msomajo(setup):
    mydict, text = setup
    mydict["somajo_dict"]["model"] = "en_PTB"
    mydict["somajo_dict"]["processors"] = "sentencize", "tokenize"
    mydict["advanced_options"]["output_dir"] = "./test/out/"
    mydict["advanced_options"]["corpus_dir"] = "./test/corpora/"
    mydict["advanced_options"]["registry_dir"] = "./test/registry/"
    tokenized = mso.MySomajo(mydict["somajo_dict"])
    tokenized.apply_to(text)
    # we should not need start ..?
    start = 0
    # for somajo we never have list data as this will be only used for sentencizing
    out_obj = mso.OutSomajo(tokenized.doc, tokenized.jobs, start, islist=False)
    out = out_obj.assemble_output_sent()
    stags = out_obj.get_stags()
    ptags = None
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    be.OutObject.write_vrt(outfile, out)
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)


def test_integration_mtreetagger(setup):

    out = TemporaryDirectory()

    mydict, text = setup
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)
