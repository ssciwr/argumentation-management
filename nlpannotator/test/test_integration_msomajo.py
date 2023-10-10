import pytest
import nlpannotator.base as be
import nlpannotator.msomajo as mso
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


@pytest.fixture
def setup():
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    input_file = pkg / "data" / "example_en.txt"
    mydict["input"] = input_file.as_posix()
    be.PrepareRun.validate_input_dict(mydict)
    text = be.PrepareRun.get_text(mydict["input"])

    return mydict, text


def test_integration_msomajo(setup, tmp_path):
    mydict, text = setup
    mydict["somajo_dict"]["model"] = "en_PTB"
    mydict["somajo_dict"]["processors"] = "sentencize", "tokenize"
    outdir = tmp_path / "out"
    corpusdir = tmp_path / "corpora"
    registrydir = tmp_path / "registry"
    mydict["advanced_options"]["output_dir"] = outdir.as_posix()
    mydict["advanced_options"]["corpus_dir"] = corpusdir.as_posix()
    mydict["advanced_options"]["registry_dir"] = registrydir.as_posix()
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
    encode_obj = be.EncodeCorpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
