import pytest
import nlpannotator.base as be
import nlpannotator.pipe as pe
import nlpannotator.mspacy as msp
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


@pytest.fixture()
def load_data():
    data = be.PrepareRun.get_text(pkg / "data" / "example_de.txt")
    return data


def test_integration_mspacy(load_data, tmp_path):
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    mydict["language"] = "en"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "fast"
    mydict["processing_type"] = "sentencize, tokenize, pos, lemma"
    inputfile = pkg / "data" / "example_en.txt"
    mydict["input"] = inputfile.as_posix()
    outdir = tmp_path / "out"
    corpusdir = tmp_path / "corpora"
    registrydir = tmp_path / "registry"
    mydict["advanced_options"]["output_dir"] = outdir.as_posix()
    mydict["advanced_options"]["corpus_dir"] = corpusdir.as_posix()
    mydict["advanced_options"]["registry_dir"] = registrydir.as_posix()
    be.PrepareRun.validate_input_dict(mydict)
    obj = pe.SetConfig(mydict)
    spacy_dict = obj.mydict["spacy_dict"]
    annotated = msp.MySpacy(spacy_dict)
    data = load_data
    # apply pipeline to data
    annotated.apply_to(data)
    start = 0
    out_obj = msp.OutSpacy(annotated.doc, annotated.jobs, start=start)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.ptags
    stags = out_obj.stags
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    encode_obj = be.EncodeCorpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
