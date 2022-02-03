from json import load
import pytest
import spacy as sp
import mspacy as msp
import base as be
import tempfile


@pytest.fixture()
def init():
    """Load the input dicts"""

    mydict = be.prepare_run.load_input_dict("test/test_files/input")
    mydict_test = be.prepare_run.load_input_dict("test/test_files/input_short")
    return mydict, mydict_test


@pytest.fixture()
def load_object(init):
    """Initialize the test object"""

    test_obj = msp.spacy_pipe(init[0])
    return test_obj


def test_init(init, load_object):
    """Check if the parameters from the input dict are loaded into the
    pipe object as expected."""

    mydict, mydict_test = init
    test_obj = load_object
    assert test_obj.outname == mydict["output"]
    assert test_obj.pretrained == mydict_test["pretrained"]
    assert test_obj.lang == mydict_test["lang"]
    assert test_obj.type == mydict_test["text_type"]
    assert test_obj.model == mydict_test["model"]
    assert test_obj.jobs == [
        proc.strip() for proc in mydict_test["processors"].split(",")
    ]
    assert test_obj.config == be.prepare_run.update_dict(mydict_test["config"])


@pytest.fixture()
def pipe_sent(init, load_object):
    """Check if applying pipeline through mspacy leads to same result as applying
    same pipeline through spacy directly."""

    text = """This is an example text. This is a second sentence."""

    test_obj = load_object
    mydict = init[1]
    # as this pipe config should just load all of a model, results
    # should be equivalent to using:
    nlp = sp.load(mydict["model"])
    check_doc = nlp(text)
    test_doc = test_obj.apply_to(text).doc
    assert test_doc.has_annotation("SENT_START")
    # there seems to be no __eq__() definition for either doc or token in spacy
    # so just compare a couple of attributes for every token?
    for test_token, token in zip(test_doc, check_doc):
        assert test_token.text == token.text
        assert test_token.lemma == token.lemma
        assert test_token.pos == token.pos
        assert test_token.tag == token.tag
        assert test_token.sent_start == token.sent_start
    return test_obj.apply_to(text), check_doc


def test_output_sent(pipe_sent):
    """Check if output is as expected, use current output as example result.
    Additionally use doc build through spacy directly and compare output."""

    check = [
        "<s>\n",
        "This\tDT\tthis\t-\tnsubj\tPRON\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "an\tDT\tan\t-\tdet\tDET\n",
        "example\tNN\texample\t-\tcompound\tNOUN\n",
        "text\tNN\ttext\t-\tattr\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
        "<s>\n",
        "This\tDT\tthis\t-\tnsubj\tPRON\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "a\tDT\ta\t-\tdet\tDET\n",
        "second\tJJ\tsecond\tORDINAL\tamod\tADJ\n",
        "sentence\tNN\tsentence\t-\tattr\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
    ]
    # this is quite specific, any way to generalize?
    test_obj, check_doc = pipe_sent
    test_out = msp.out_object_spacy(test_obj.doc, test_obj.jobs, start=0).fetch_output()
    check_out = msp.out_object_spacy(check_doc, test_obj.jobs, start=0).fetch_output()

    assert test_out == check_out
    assert test_out == check


@pytest.fixture()
def chunked_data():
    text = '<textid="1"> This is an example text. <subtextid="1"> It has some subtext. </subtext> </text> <textid="2"> Here is some more text. </text>'
    formated_text = text.replace(" ", "\n")

    tmp = tempfile.NamedTemporaryFile()

    tmp.write(formated_text.encode())
    tmp.seek(0)
    # print(tmp.read().decode())
    data = be.chunk_sample_text("{}".format(tmp.name))
    # print(data)
    # don't need this anymore
    tmp.close()

    return data


def test_pipe_multiple(load_object, chunked_data):
    """Check if the pipe_multiple function works correctly."""

    test_obj = load_object

    # lets just quickly emulate a file for our input, maybe change the chunker to also allow for direct string input down the line?
    data = chunked_data
    results_pipe = test_obj.pipe_multiple(data, ret=True)
    results_alt = test_obj.get_multiple(data, ret=True)

    # using spacy 3.2.1 and en_core_web_md 3.2.0
    check_chunked = [
        '<textid="1"> \n',
        "<s>\n",
        "This\tDT\tthis\t-\tnsubj\tPRON\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "an\tDT\tan\t-\tdet\tDET\n",
        "example\tNN\texample\t-\tcompound\tNOUN\n",
        "text\tNN\ttext\t-\tattr\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
        "\n",
        '<subtextid="1"> \n',
        "<s>\n",
        "It\tPRP\tit\t-\tnsubj\tPRON\n",
        "has\tVBZ\thave\t-\tROOT\tVERB\n",
        "some\tDT\tsome\t-\tdet\tDET\n",
        "subtext\tNN\tsubtext\t-\tdobj\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
        "</subtext> \n",
        "\n",
        "</text> \n",
        '<textid="2"> \n',
        "<s>\n",
        "Here\tRB\there\t-\tadvmod\tADV\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "some\tDT\tsome\t-\tadvmod\tPRON\n",
        "more\tJJR\tmore\t-\tamod\tADJ\n",
        "text\tNN\ttext\t-\tnsubj\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
        "</text>\n",
    ]

    assert type(results_pipe) == list
    assert check_chunked == results_pipe
    assert check_chunked == results_alt


def test_sentencize():

    text_en = "This is a sentence. This is another sentence, or is it?"
    text_de = "Dies ist ein Satz. Dies ist ein zweiter Satz, oder nicht?"

    data_en = msp.sentencize_spacy("en", text_en)
    data_de = msp.sentencize_spacy("de", text_de)

    check_en = [["This is a sentence.", 4], ["This is another sentence, or is it?", 11]]
    check_de = [
        ["Dies ist ein Satz.", 4],
        ["Dies ist ein zweiter Satz, oder nicht?", 11],
    ]

    assert data_en == check_en
    assert data_de == check_de
