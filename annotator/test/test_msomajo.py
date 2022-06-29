import pytest
import base as be
import msomajo as mso

test_out_en = [
    "<s>\n",
    "The\n",
    "Scientific\n",
    "Software\n",
    "Center\n",
    "strives\n",
    "to\n",
    "improve\n",
    "scientific\n",
    "software\n",
    "development\n",
    "to\n",
    "ensure\n",
    "reproducible\n",
    "science\n",
    "and\n",
    "software\n",
    "sustainability\n",
    ".\n",
    "</s>\n",
    "<s>\n",
    "The\n",
    "SSC\n",
    "also\n",
    "acts\n",
    "as\n",
    "a\n",
    "link\n",
    "between\n",
    "the\n",
    "different\n",
    "scientific\n",
    "disciplines\n",
    ",\n",
    "enabling\n",
    "collaboration\n",
    "and\n",
    "interdisciplinary\n",
    "research\n",
    ".\n",
    "</s>\n",
    "<s>\n",
    "The\n",
    "current\n",
    "role\n",
    "of\n",
    "software\n",
    "in\n",
    "research\n",
    "communities\n",
    "Software\n",
    "development\n",
    "is\n",
    "an\n",
    "increasingly\n",
    "vital\n",
    "part\n",
    "of\n",
    "research\n",
    ",\n",
    "but\n",
    "if\n",
    "not\n",
    "done\n",
    "sustainably\n",
    "the\n",
    "result\n",
    "is\n",
    "often\n",
    "unmaintainable\n",
    "software\n",
    "and\n",
    "irreproducible\n",
    "science\n",
    ".\n",
    "</s>\n",
    "<s>\n",
    "This\n",
    "is\n",
    "due\n",
    "to\n",
    "a\n",
    "lack\n",
    "of\n",
    "software\n",
    "engineering\n",
    "training\n",
    "for\n",
    "scientists\n",
    ",\n",
    "limited\n",
    "funding\n",
    "for\n",
    "maintaining\n",
    "existing\n",
    "software\n",
    "and\n",
    "few\n",
    "permanent\n",
    "software\n",
    "developer\n",
    "positions\n",
    ".\n",
    "</s>\n",
    "<s>\n",
    "The\n",
    "SSC\n",
    "addresses\n",
    "the\n",
    "current\n",
    "shortcomings\n",
    "by\n",
    "implementing\n",
    "the\n",
    "three\n",
    "pillars\n",
    "of\n",
    "Development\n",
    ",\n",
    "Teaching\n",
    "and\n",
    "Outreach\n",
    ".\n",
    "</s>\n",
]
test_out_en_sentence = [
    "The\n",
    "Scientific\n",
    "Software\n",
    "Center\n",
    "strives\n",
    "to\n",
    "improve\n",
    "scientific\n",
    "software\n",
    "development\n",
    "to\n",
    "ensure\n",
    "reproducible\n",
    "science\n",
    "and\n",
    "software\n",
    "sustainability\n",
    ".\n",
]
test_out_en_sentences = [
    "The Scientific Software Center strives to improve scientific software development to ensure reproducible science and software sustainability .",
    "The SSC also acts as a link between the different scientific disciplines , enabling collaboration and interdisciplinary research .",
    "The current role of software in research communities Software development is an increasingly vital part of research , but if not done sustainably the result is often unmaintainable software and irreproducible science .",
    "This is due to a lack of software engineering training for scientists , limited funding for maintaining existing software and few permanent software developer positions .",
    "The SSC addresses the current shortcomings by implementing the three pillars of Development , Teaching and Outreach .",
]


@pytest.fixture
def read_data_en():
    return be.PrepareRun.get_text("test/test_files/example_en.txt")


@pytest.fixture
def read_data_de():
    return be.PrepareRun.get_text("test/test_files/example_de.txt")


@pytest.fixture
def read_test_de():
    with open("test/test_files/example_de_somajo.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def read_test_en():
    with open("test/test_files/example_en_somajo.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def load_dict():
    mydict = be.PrepareRun.load_input_dict("./test/test_files/input")
    mydict["somajo_dict"]["model"] = "en_PTB"
    mydict["somajo_dict"]["processors"] = "sentencize", "tokenize"
    return mydict


@pytest.fixture
def get_doc(read_data_en, load_dict):
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    tokenized.apply_to(read_data_en)
    return tokenized.doc, tokenized.jobs


def test_mysomajo_init(load_dict):
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    assert tokenized.jobs == ("sentencize", "tokenize")
    assert tokenized.model == "en_PTB"
    assert tokenized.sentencize
    assert tokenized.camelcase


def test_apply_to(read_data_en, read_data_de, load_dict):
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    tokenized.apply_to(read_data_en)
    assert tokenized.doc[0][8].text == "software"
    load_dict["somajo_dict"]["model"] = "de_CMC"
    tokenized = mso.MySomajo(load_dict["somajo_dict"])
    tokenized.apply_to(read_data_de)
    assert tokenized.doc[2][5].text == "dass"


def test_outsomajo_init(get_doc):
    out_obj = mso.OutSomajo(get_doc[0], get_doc[1], 0, islist=False)
    assert out_obj.attrnames["proc_sent"] == "sentencize"
    assert out_obj.attrnames["sentence"] == "sent"
    assert out_obj.stags == ["s"]
    assert not out_obj.ptags


def test_assemble_output_sent(get_doc):
    out_obj = mso.OutSomajo(get_doc[0], get_doc[1], 0, islist=False)
    out = out_obj.assemble_output_sent()
    assert out == test_out_en


def test_iterate(get_doc):
    out_obj = mso.OutSomajo(get_doc[0], get_doc[1], 0, islist=False)
    out = []
    sent = get_doc[0][0]
    out_obj.iterate(out, sent, "STR")
    assert out == test_out_en_sentence


def test_sentences(get_doc):
    out_obj = mso.OutSomajo(get_doc[0], get_doc[1], 0, islist=False)
    assert out_obj.sentences == test_out_en_sentences
