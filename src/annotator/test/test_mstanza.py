import pytest
import annotator.mstanza as ma


@pytest.fixture()
def get_sample(request):
    """Load the specified sample text."""
    marker = request.node.get_closest_marker("lang")
    if marker == "None":
        # missing marker
        print("Missing a marker for reading the sample text.")
    else:
        lang_key = marker.args[0]
    text_dict = {
        "en": "./test/test_files/example_en.txt",
        "de": "./test/test_files/example_de.txt",
        "test_en": "./test/test_files/example_en_stanza.txt",
        "test_de": "./test/test_files/example_de_stanza.txt",
    }
    # Read the sample text.
    with open(text_dict[lang_key], "r") as myfile:
        data = myfile.read().replace("\n", "")
    # Read the processed text from stanza.
    with open(text_dict["test_" + lang_key], "r") as myfile:
        test_data = myfile.read().replace("\n", "")
    return data, test_data


test_doc = """[\n  [\n    {\n      "id": 1,\n      "text": "This",\n      "lemma": "this",\n      "upos": "PRON",\n      "xpos": "DT",\n      "feats": "Number=Sing|PronType=Dem",\n      "start_char": 0,\n      "end_char": 4\n    },\n    {\n      "id": 2,\n      "text": "is",\n      "lemma": "be",\n      "upos": "AUX",\n      "xpos": "VBZ",\n      "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",\n      "start_char": 5,\n      "end_char": 7\n    },\n    {\n      "id": 3,\n      "text": "an",\n      "lemma": "a",\n      "upos": "DET",\n      "xpos": "DT",\n      "feats": "Definite=Ind|PronType=Art",\n      "start_char": 8,\n      "end_char": 10\n    },\n    {\n      "id": 4,\n      "text": "example",\n      "lemma": "example",\n      "upos": "NOUN",\n      "xpos": "NN",\n      "feats": "Number=Sing",\n      "start_char": 11,\n      "end_char": 18\n    },\n    {\n      "id": 5,\n      "text": ".",\n      "lemma": ".",\n      "upos": "PUNCT",\n      "xpos": ".",\n      "start_char": 18,\n      "end_char": 19\n    }\n  ],\n  [\n    {\n      "id": 1,\n      "text": "And",\n      "lemma": "and",\n      "upos": "CCONJ",\n      "xpos": "CC",\n      "start_char": 20,\n      "end_char": 23\n    },\n    {\n      "id": 2,\n      "text": "here",\n      "lemma": "here",\n      "upos": "ADV",\n      "xpos": "RB",\n      "feats": "PronType=Dem",\n      "start_char": 24,\n      "end_char": 28\n    },\n    {\n      "id": 3,\n      "text": "we",\n      "lemma": "we",\n      "upos": "PRON",\n      "xpos": "PRP",\n      "feats": "Case=Nom|Number=Plur|Person=1|PronType=Prs",\n      "start_char": 29,\n      "end_char": 31\n    },\n    {\n      "id": 4,\n      "text": "go",\n      "lemma": "go",\n      "upos": "VERB",\n      "xpos": "VBP",\n      "feats": "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin",\n      "start_char": 32,\n      "end_char": 34\n    },\n    {\n      "id": 5,\n      "text": ".",\n      "lemma": ".",\n      "upos": "PUNCT",\n      "xpos": ".",\n      "start_char": 34,\n      "end_char": 35\n    }\n  ]\n]"""


def test_fix_dict_path():
    mydict = {
        "dir": "./test/models",
        "tokenize_model_path": "en/tokenize/combined.pt",
    }
    mydict = ma.mstanza_preprocess.fix_dict_path(mydict)
    test_mydict = {
        "dir": "./test/models",
        "tokenize_model_path": "./test/models/en/tokenize/combined.pt",
    }
    assert mydict == test_mydict


def test_init_pipeline():
    mydict = {
        "lang": "en",
        "dir": "./test/models/",
        "processors": "tokenize,pos,lemma",
    }
    obj = ma.mstanza_pipeline(mydict)
    obj.init_pipeline()


@pytest.mark.lang("en")
def test_process_text_en(get_sample):
    mydict = {
        "lang": "en",
        "dir": "./test/models/",
        "processors": "tokenize,pos,lemma",
    }
    text, test_doc = get_sample
    obj = ma.mstanza_pipeline(mydict)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    doc = str(docobj).replace("\n", "")
    assert doc == str(test_doc)


@pytest.mark.lang("de")
def test_process_text_de(get_sample):
    mydict = {
        "lang": "de",
        "dir": "./test/models/",
        "processors": "tokenize,pos,lemma",
    }
    text, test_doc = get_sample
    obj = ma.mstanza_pipeline(mydict)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    doc = str(docobj).replace("\n", "")
    assert doc == str(test_doc)
