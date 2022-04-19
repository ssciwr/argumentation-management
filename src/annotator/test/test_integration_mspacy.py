import pytest
import base as be
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
    mydict["tool"] = "spacy"
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)
    be.prepare_run.validate_input_dict(mydict)
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    data = load_data
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results(mydict)
