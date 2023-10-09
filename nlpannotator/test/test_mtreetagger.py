import pytest
import nlpannotator.base as be
import nlpannotator.mtreetagger as mtt
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


@pytest.mark.treetagger
@pytest.fixture
def data_en():
    return "This is a sentence."


@pytest.mark.treetagger
@pytest.fixture
def test_en():
    data = [
        "<s>",
        "This\tDT\tthis\n",
        "is\tVBZ\tbe\n",
        "a\tDT\ta\n",
        "sentence\tNN\tsentence\n",
        ".\tSENT\t.\n",
        "</s>",
    ]
    return data


@pytest.mark.treetagger
@pytest.fixture
def test_dict_doc():
    dict_doc = [
        {"word": "This", "pos": "DT", "lemma": "this"},
        {"word": "is", "pos": "VBZ", "lemma": "be"},
        {"word": "a", "pos": "DT", "lemma": "a"},
        {"word": "sentence", "pos": "NN", "lemma": "sentence"},
        {"word": ".", "pos": "SENT", "lemma": "."},
    ]
    return dict_doc


@pytest.mark.treetagger
@pytest.fixture
def load_dict():
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    mydict["treetagger_dict"]["lang"] = "en"
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    return mydict["treetagger_dict"]


@pytest.mark.treetagger
@pytest.fixture
def get_doc(load_dict, data_en):
    annotated = mtt.MyTreetagger(load_dict)
    annotated.apply_to(data_en)
    return annotated.doc, annotated.jobs


@pytest.mark.treetagger
def test_mytreetagger_init(load_dict):
    annotated = mtt.MyTreetagger(load_dict)
    assert annotated.jobs == ("tokenize", "pos", "lemma")


@pytest.mark.treetagger
def test_mytreetagger_apply_to(get_doc):
    assert get_doc[0][0].text == "This"
    assert get_doc[0][0].pos == "DT"
    assert get_doc[0][0].lemma == "this"
    assert get_doc[0][3].text == "sentence"
    assert get_doc[0][3].pos == "NN"
    assert get_doc[0][3].lemma == "sentence"


@pytest.mark.treetagger
def test_mytreetagger_make_dict(load_dict, data_en, test_dict_doc):
    annotated = mtt.MyTreetagger(load_dict)
    annotated.doc = annotated.nlp.tag_text(data_en)
    annotated.doc = annotated._make_dict()
    assert annotated.doc == test_dict_doc


@pytest.mark.treetagger
def test_mytreetagger_make_object(load_dict, data_en):
    annotated = mtt.MyTreetagger(load_dict)
    annotated.doc = annotated.nlp.tag_text(data_en)
    annotated.doc = annotated._make_dict()
    annotated._make_object()
    assert annotated.doc[1].text == "is"
    assert annotated.doc[1].pos == "VBZ"
    assert annotated.doc[1].lemma == "be"


@pytest.mark.treetagger
def test_outtreetagger_init(get_doc):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    assert out_obj.attrnames["proc_sent"] == "na"
    assert out_obj.attrnames["proc_lemma"] == "lemma"
    assert not out_obj.stags


@pytest.mark.treetagger
def test_assemble_output_tokens(get_doc, test_en):
    out_obj = mtt.OutTreetagger(get_doc[0], get_doc[1], 0)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    assert out == test_en
