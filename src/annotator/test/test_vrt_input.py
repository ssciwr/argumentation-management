import pytest
import annotator.base as be
import annotator.mspacy as msp
import annotator.mstanza as mst


@pytest.fixture
def grab_data():
    minimal_example = be.prepare_run.load_tokens_from_vrt(
        "test/test_files/example_pretokenized.vrt"
    )
    return minimal_example


@pytest.fixture
def setup():
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    be.prepare_run.validate_input_dict(mydict)
    return mydict


def test_spacy_pretokenized(setup, grab_data):

    data = grab_data
    mydict = setup
    pipe = msp.spacy_pipe(mydict["spacy_dict"])
    annotated = pipe.apply_to(data, pretokenized=True)
    out = annotated.pass_results("STR", mydict, ret=True)

    check = [
        "<s>\n",
        "This\tDT\tthis\t \tnsubj\tPRON\n",
        "is\tVBZ\tbe\t \tROOT\tAUX\n",
        "an\tDT\tan\t \tdet\tDET\n",
        "example\tNN\texample\t \tcompound\tNOUN\n",
        "text\tNN\ttext\t \tattr\tNOUN\n",
        ".\t.\t.\t \tpunct\tPUNCT\n",
        "</s>\n",
        "<s>\n",
        "This\tDT\tthis\t \tnsubj\tPRON\n",
        "is\tVBZ\tbe\t \tROOT\tAUX\n",
        "a\tDT\ta\t \tdet\tDET\n",
        "second\tJJ\tsecond\tORDINAL\tamod\tADJ\n",
        "sentence\tNN\tsentence\t \tattr\tNOUN\n",
        ".\t.\t.\t \tpunct\tPUNCT\n",
        "</s>\n",
    ]

    assert out == check


def test_stanza_pretokenized(setup, grab_data):

    data = grab_data
    mydict = setup
    pipe = mst.MyStanza(mydict["stanza_dict"], pretokenized=True)
    out = pipe.apply_to(data).pass_results(mydict, ret=True)

    check = [
        "<s>\n",
        "This\tPRON\tthis\n",
        "is\tAUX\tbe\n",
        "an\tDET\ta\n",
        "example\tNOUN\texample\n",
        "text\tNOUN\ttext\n",
        ".\tPUNCT\t.\n",
        "</s>\n",
        "<s>\n",
        "This\tPRON\tthis\n",
        "is\tAUX\tbe\n",
        "a\tDET\ta\n",
        "second\tADJ\tsecond\n",
        "sentence\tNOUN\tsentence\n",
        ".\tPUNCT\t.\n",
        "</s>\n",
    ]

    assert out == check
