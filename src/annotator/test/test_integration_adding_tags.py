import pytest
import base as be
import mtreetagger as mtt
import mspacy as msp
from tempfile import TemporaryDirectory
import os


@pytest.fixture()
def load_data():
    data = be.prepare_run.get_text("./test/test_files/example_en.txt")
    return data


def test_integration_adding_tags(load_data):

    out = TemporaryDirectory()
    corp = TemporaryDirectory()
    reg = TemporaryDirectory()
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)
    mydict["advanced_options"]["corpus_dir"] = "{}".format(corp.name)
    mydict["advanced_options"]["registry_dir"] = "{}".format(reg.name)

    data = load_data
    treetagger_dict = mydict["treetagger_dict"]
    pipe = mtt.treetagger_pipe(treetagger_dict)

    out = pipe.apply_to(data)

    out.pass_results(mydict, "STR")
    cwd = os.getcwd()
    os.system("cd {} && touch test && cd {}".format(reg.name, cwd))
    mydict["tool"] = "spacy"
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results(
        mydict, add=True, ptags=["test1", "test2", "test3", "test4", "test5"]
    )
