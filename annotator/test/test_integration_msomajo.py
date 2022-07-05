import pytest
import base as be
import msomajo as mso


@pytest.fixture
def setup():
    mydict = be.PrepareRun.load_input_dict("input")
    mydict["input"] = "./test/data/example_en.txt"
    be.PrepareRun.validate_input_dict(mydict)
    text = be.PrepareRun.get_text(mydict["input"])

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
    out_obj = mso.OutSomajo(tokenized.doc, tokenized.jobs, start)
    out = out_obj.assemble_output_sent()
    stags = out_obj.get_stags()
    ptags = None
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    be.OutObject.write_vrt(outfile, out)
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
