import pytest

# from .context import base as be
import base as be

# from .context import mtreetagger as mtt
import mtreetagger as mtt

# from .context import mspacy as msp
import mspacy as msp
from tempfile import TemporaryDirectory
import os


@pytest.fixture()
def load_data():
    data = "This is a sentence."
    return data


def test_integration_adding_tags(load_data):

    out = TemporaryDirectory()
    corp = TemporaryDirectory()
    reg = TemporaryDirectory()
    mydict = be.prepare_run.load_input_dict("input")
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)
    mydict["advanced_options"]["corpus_dir"] = "{}".format(corp.name)
    mydict["advanced_options"]["registry_dir"] = "{}".format(reg.name)
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    data = load_data
    treetagger_dict = mydict["treetagger_dict"]
    annotated = mtt.MyTreetagger(treetagger_dict)

    annotated = annotated.apply_to(data)
    start = 0
    out_obj = mtt.OutTreetagger(
        annotated.doc, annotated.jobs, start=start, islist=False
    )
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)

    cwd = os.getcwd()
    os.system("cd {} && touch test && cd {}".format(reg.name, cwd))
    mydict["tool"] = "spacy"
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    # annotated = msp.MySpacy(spacy_dict)
    # apply pipeline to data
    # annotated.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    # this is not working - first purges old corpus and then tries to add to it
    # of course then the dir + vrt is missing
    # annotated.pass_results(mydict, add=True)
