import pytest
import mflair as mf
import base as be
import tempfile


@pytest.fixture()
def init():
    return be.prepare_run.load_input_dict("test/test_files/input")


@pytest.fixture()
def load_object(init):
    # init defined in test_mspacy
    return mf.flair_pipe(init)


@pytest.fixture
def apply_to(load_object):
    """Get object in post-pipe state."""

    test_obj = load_object

    post_pipe = test_obj.senter_spacy(text_en).apply_to()

    return post_pipe


@pytest.fixture()
def chunked_data():
    """Get some chunked data."""

    text = '<textid="1"> This is an example text. <subtextid="1"> It has some subtext. </subtext> </text> <textid="2"> Here is some more text. </text>'
    formated_text = text.replace(" ", "\n")

    tmp = tempfile.NamedTemporaryFile()

    tmp.write(formated_text.encode())
    tmp.seek(0)

    data = be.chunk_sample_text("{}".format(tmp.name))
    # don't need this anymore
    tmp.close()

    return data


def test_init(load_object, init):
    """Check that the initial values are as expected."""

    test_obj = load_object
    check_dict = init

    assert (
        test_obj.outname
        == check_dict["advanced_options"]["output_dir"] + check_dict["corpus_name"]
    )
    assert test_obj.input == check_dict["input"]
    assert test_obj.lang == check_dict["language"]
    assert test_obj.job == check_dict["flair_dict"]["job"]


text_en = "This is a sentence. This is another sentence, or is it?"


def test_senter_spacy(load_object):
    """Check that the sentencizer works."""

    # using spacy 3.2.1
    test_obj = load_object
    data_en = test_obj.senter_spacy(text_en).sents

    check_en = [["This is a sentence.", 4], ["This is another sentence, or is it?", 11]]
    assert data_en == check_en


def test_output(apply_to):
    """Check that the output is as expected."""

    # using flair 0.9
    check = """! ner pos
    <s>
    This -  DT
    is -  VBZ
    a -  DT
    sentence -  NN
    . -  .
    </s>
    <s>
    This -  DT
    is -  VBZ
    another -  DT
    sentence -  NN
    , -  ,
    or -  CC
    is -  VBZ
    it -  PRP
    ? -  .
    </s>"""
    checklist = [string.strip() + "\n" for string in check.split("\n")]
    data_pipe = apply_to.get_out(ret=True)

    assert data_pipe == checklist


def test_get_multiple(chunked_data, load_object):
    """Check that chunked data is annotated and returned as expected."""

    test_obj = load_object
    data = test_obj.get_multiple(chunked_data, ret=True)

    # using spacy 3.2.1 and flair 0.9
    check_chunked = [
        "! ner pos\n",
        '<textid="1"> \n',
        "This -  DT\n",
        "is -  VBZ\n",
        "an -  DT\n",
        "example -  NN\n",
        "text -  NN\n",
        ". -  .\n",
        '<subtextid="1"> \n',
        "It -  PRP\n",
        "has -  VBZ\n",
        "some -  DT\n",
        "subtext -  NN\n",
        ". -  .\n",
        "</subtext> \n",
        "</text> \n",
        '<textid="2"> \n',
        "Here -  RB\n",
        "is -  VBZ\n",
        "some -  DT\n",
        "more -  JJR\n",
        "text -  NN\n",
        ". -  .\n",
        "</text>\n",
    ]

    assert data == check_chunked
