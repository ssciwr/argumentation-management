import pytest
import spacy as sp
from .context import base as be
from .context import mspacy as msp
import tempfile


@pytest.fixture(scope="module")
def init():
    """Load the input dicts"""

    mydict = be.prepare_run.load_input_dict("test/test_files/input")

    subdict_test = be.prepare_run.load_input_dict("test/test_files/input_short")
    return mydict["spacy_dict"], subdict_test, mydict


@pytest.fixture(scope="module")
def load_object(init):
    """Initialize the test object"""

    init[0]["processors"] = [
        "tok2vec",
        "tagger",
        "senter",
        "parser",
        "attribute_ruler",
        "lemmatizer",
        "ner",
    ]
    test_obj = msp.MySpacy(init[0])
    return test_obj


@pytest.fixture(scope="module")
def get_text():
    """Get the sample text for testing."""
    text = """This is an example text. This is a second sentence."""
    return text


@pytest.fixture()
def pipe_sent(init, load_object, get_text):
    """Check if applying pipeline through mspacy leads to same result as applying
    same pipeline through spacy directly."""

    test_obj = load_object
    out = test_obj.apply_to(get_text)
    mydict = init[1]
    # as this pipe config should just load all of a model, results
    # should be equivalent to using:
    nlp = sp.load(mydict["model"])
    check_doc = nlp(get_text)
    test_doc = test_obj.apply_to(get_text).doc
    return test_obj.apply_to(get_text), check_doc, test_doc


@pytest.fixture()
def chunked_data():
    text = '<textid="1"> This is an example text. <subtextid="1"> It has some subtext. </subtext> </text> <textid="2"> Here is some more text. </text>'
    formated_text = text.replace(" ", "\n")

    tmp = tempfile.NamedTemporaryFile()

    tmp.write(formated_text.encode())
    tmp.seek(0)
    # print(tmp.read().decode())
    data = be.chunk_sample_text("{}".format(tmp.name))
    tmp.close()

    return data


def test_pipe(pipe_sent):
    _, check_doc, test_doc = pipe_sent

    assert test_doc.has_annotation("SENT_START")
    # there seems to be no __eq__() definition for either doc or token in spacy
    # so just compare a couple of attributes for every token?
    for test_token, token in zip(test_doc, check_doc):
        assert test_token.text == token.text
        assert test_token.lemma == token.lemma
        assert test_token.pos == token.pos
        assert test_token.tag == token.tag
        assert test_token.sent_start == token.sent_start


def test_init():
    dictl = [
        {
            "model": "en_core_web_md",
            "lang": "en",
            "processors": ["lemmatizer", "tagger"],
            "set_device": "prefer_GPU",
            "config": {},
        },
        {
            "model": "de_core_news_md",
            "lang": "de",
            "processors": ["tagger"],
            "set_device": False,
            "config": {},
        },
        {
            "model": "de_core_news_md",
            "lang": "de",
            "processors": ["lemmatizer"],
            "set_device": False,
            "config": {},
        },
        {
            "model": "en_core_web_md",
            "lang": "en",
            "processors": ["tok2vec", "lemmatizer", "tagger"],
            "set_device": "prefer_GPU",
            "config": {},
        },
    ]

    modell = ["en_core_web_md", "de_core_news_md", "de_core_news_md", "en_core_web_md"]
    processors = [
        ["tok2vec", "lemmatizer", "tagger", "attribute_ruler"],
        ["tok2vec", "tagger", "attribute_ruler"],
        ["tok2vec", "lemmatizer", "attribute_ruler"],
        ["tok2vec", "lemmatizer", "tagger", "attribute_ruler"],
    ]
    # should also check for CPU/GPU here
    for subdict, model, procs in zip(dictl, modell, processors):
        test_obj = msp.MySpacy(subdict)
        assert test_obj.model == model
        assert test_obj.jobs == procs


def test_init_pipe(init, load_object):
    """Check if the parameters from the input dict are loaded into the
    pipe object as expected."""

    mydict_test = init[1]
    test_obj = load_object

    assert test_obj.model == mydict_test["model"]
    assert test_obj.jobs == mydict_test["processors"]
    assert test_obj.config == mydict_test["config"]


def test_apply_to(load_object, get_text):
    test_obj = load_object.apply_to(get_text)
    assert str(test_obj.doc) == get_text


def test_output_sent(pipe_sent):
    """Check if output is as expected, use current output as example result.
    Additionally use doc build through spacy directly and compare output."""

    check = [
        "<s>\n",
        "This\tPRON\tthis\t \n",
        "is\tAUX\tbe\t \n",
        "an\tDET\tan\t \n",
        "example\tNOUN\texample\t \n",
        "text\tNOUN\ttext\t \n",
        ".\tPUNCT\t.\t \n",
        "</s>\n",
        "<s>\n",
        "This\tPRON\tthis\t \n",
        "is\tAUX\tbe\t \n",
        "a\tDET\ta\t \n",
        "second\tADJ\tsecond\tORDINAL\n",
        "sentence\tNOUN\tsentence\t \n",
        ".\tPUNCT\t.\t \n",
        "</s>\n",
    ]
    # this is quite specific, any way to generalize?
    test_obj, check_doc, _ = pipe_sent
    test_out_obj = msp.OutSpacy(test_obj.doc, test_obj.jobs, start=0)
    check_out_obj = msp.OutSpacy(check_doc, test_obj.jobs, start=0)
    test_out = test_out_obj.assemble_output_sent()
    check_out = check_out_obj.assemble_output_sent()
    test_out = test_out_obj.assemble_output_tokens(test_out)
    check_out = check_out_obj.assemble_output_tokens(check_out)
    print("test_out:", test_out)
    print("check_out:", check_out)
    print("check:", check)
    assert test_out == check_out
    assert test_out == check


def test_sentencize():

    text_en = "This is a sentence. This is another sentence, or is it?"
    text_de = "Dies ist ein Satz. Dies ist ein zweiter Satz, oder nicht?"

    data_en = msp.MySpacy.sentencize_spacy("en_core_web_md", text_en)
    data_de = msp.MySpacy.sentencize_spacy("de_core_news_md", text_de)

    check_en = [["This is a sentence.", 4], ["This is another sentence, or is it?", 11]]
    check_de = [
        ["Dies ist ein Satz.", 4],
        ["Dies ist ein zweiter Satz, oder nicht?", 11],
    ]

    assert data_en == check_en
    assert data_de == check_de
