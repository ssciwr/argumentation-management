import pytest
import annotator.base as be
import annotator.mflair as mf


@pytest.fixture
def data_en():
    return "This is a sentence."


@pytest.fixture
def data_en_list():
    return ["This is a sentence.", "This is another sentence."]


@pytest.fixture
def test_en():
    data = [
        "<s>",
        "This\tDT\n",
        "is\tVBZ\n",
        "a\tDT\n",
        "sentence\tNN\n",
        ".\t.\n",
        "</s>",
    ]
    return data


@pytest.fixture
def load_dict():
    mydict = be.PrepareRun.load_input_dict("./test/data/input.json")
    mydict["flair_dict"]["lang"] = "en"
    mydict["flair_dict"]["model"] = "pos"
    mydict["flair_dict"]["processors"] = ["tokenize", "pos"]
    return mydict["flair_dict"]


@pytest.fixture
def get_doc(load_dict, data_en):
    annotated = mf.MyFlair(load_dict)
    annotated.apply_to(data_en)
    return annotated.doc, annotated.jobs


labels = ["DT", "VBZ", "DT", "NN", "."]


def test_myflair_init(load_dict):
    annotated = mf.MyFlair(load_dict)
    assert annotated.jobs == ["tokenize", "pos"]
    assert annotated.model == "pos"
    assert "MultiTagger" in str(annotated.nlp)
    load_dict["processors"] = "pos"
    annotated = mf.MyFlair(load_dict)
    assert "SequenceTagger" in str(annotated.nlp)


def test_myflair_apply_to(get_doc):
    assert get_doc[0][0].text == "This"
    assert get_doc[0][0].get_label("pos").value == "DT"
    assert get_doc[0][3].text == "sentence"
    assert get_doc[0][3].get_label("pos").value == "NN"


def test_outflair_init(get_doc):
    out_obj = mf.OutFlair(get_doc[0], get_doc[1], 0)
    assert out_obj.attrnames["proc_sent"] == "na"
    assert out_obj.attrnames["proc_pos"] == "pos"
    assert not out_obj.stags


def test_assemble_output_tokens(get_doc, test_en):
    out_obj = mf.OutFlair(get_doc[0], get_doc[1], 0)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    print(out)
    assert out == test_en


def test_grab_tag(get_doc):
    out_obj = mf.OutFlair(get_doc[0], get_doc[1], 0)
    test_labels = []
    for token in get_doc[0]:
        test_labels.append(out_obj.grab_tag(token))
    assert test_labels == labels


def test_sentence_token_list(load_dict, data_en_list):
    doc = []
    annotated = mf.MyFlair(load_dict)
    for sentence in data_en_list:
        annotated.apply_to(sentence)
        doc.append(annotated.doc)
    out_obj = mf.OutFlair(doc, annotated.jobs, 0)
    out = out_obj.sentence_token_list(doc)
    assert out[0].text == "This"
    assert out[7].text == "another"
