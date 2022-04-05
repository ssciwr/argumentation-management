import pytest
import annotator.base as be
import annotator.msomajo as msm
import annotator.mtreetagger as mtt


@pytest.fixture
def setup():
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    be.prepare_run.validate_input_dict(mydict)
    text = be.prepare_run.get_text(mydict["input"])

    return mydict, text


def test_integration_msomajo(setup):
    mydict, text = setup
    be.prepare_run.pretokenize(
        text, mydict, msm.tokenize, arguments={"model": "en_PTB"}
    )


def test_integration_mtreetagger(setup):
    mydict, text = setup
    be.prepare_run.pretokenize(text, mydict, mtt.tokenize, arguments={"lang": "en"})
