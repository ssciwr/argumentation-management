import pytest
import annotator.base as be
import annotator.pipe as pe
import annotator.mspacy as msp


@pytest.fixture()
def load_data():
    data = be.PrepareRun.get_text("test/data/example_de.txt")
    return data


def test_integration_mspacy(load_data):
    mydict = be.PrepareRun.load_input_dict("data/input.json")
    mydict["language"] = "en"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "fast"
    mydict["processing_type"] = "tokenize, pos, lemma"
    mydict["input"] = "./test/data/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/out/"
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
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
