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
    out = TemporaryDirectory()
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["language"] = "en"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "fast"
    mydict["processing_type"] = "tokenize, pos, lemma"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    be.prepare_run.validate_input_dict(mydict)
    # load the pipe object for updating dict with settings
    obj = pe.SetConfig(mydict)
    spacy_dict = obj.mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    data = load_data
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results(mydict)
