import pytest
import base as be
import pipe as pe
import mspacy as msp
from tempfile import TemporaryDirectory


@pytest.fixture()
def load_data():
    data = be.prepare_run.get_text("./test/test_files/example_de.txt")
    return data


def test_integration_mspacy(load_data):

    # create temporary directories for the corpora
    # out = TemporaryDirectory()
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["language"] = "en"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "fast"
    mydict["processing_type"] = "tokenize, pos, lemma"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/out/"
    be.prepare_run.validate_input_dict(mydict)
    # load the pipe object for updating dict with settings
    obj = pe.SetConfig(mydict)
    spacy_dict = obj.mydict["spacy_dict"]
    # load the pipeline from the config
    annotated = msp.MySpacy(spacy_dict)
    data = load_data
    # apply pipeline to data
    annotated.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    start = 0
    out_obj = msp.OutSpacy(annotated.doc, annotated.jobs, start=start)
    style = "STR"
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.ptags
    stags = out_obj.stags
    # write to file -> This overwrites any existing file of given name;
    # as all of this should be handled internally and the files are only
    # temporary, this should not be a problem. right?
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    # add = False
    # if ret is False and style == "STR" and mydict is not None and add is False:
    # if not add:
    out_obj.write_vrt(outfile, out)
    # encode
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
    # elif ret is False and style == "STR" and mydict is not None and add is True:
    # else:
    #     out_obj.write_vrt(outfile, out)
    #     encode_obj = be.encode_corpus(mydict)
    #     encode_obj.encode_vrt(ptags, stags)
