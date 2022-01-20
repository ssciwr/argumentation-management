import pytest
from .. import mstanza as ma

test_doc = """[\n  [\n    {\n      "id": 1,\n      "text": "This",\n      "lemma": "this",\n      "upos": "PRON",\n      "xpos": "DT",\n      "feats": "Number=Sing|PronType=Dem",\n      "start_char": 0,\n      "end_char": 4\n    },\n    {\n      "id": 2,\n      "text": "is",\n      "lemma": "be",\n      "upos": "AUX",\n      "xpos": "VBZ",\n      "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",\n      "start_char": 5,\n      "end_char": 7\n    },\n    {\n      "id": 3,\n      "text": "an",\n      "lemma": "a",\n      "upos": "DET",\n      "xpos": "DT",\n      "feats": "Definite=Ind|PronType=Art",\n      "start_char": 8,\n      "end_char": 10\n    },\n    {\n      "id": 4,\n      "text": "example",\n      "lemma": "example",\n      "upos": "NOUN",\n      "xpos": "NN",\n      "feats": "Number=Sing",\n      "start_char": 11,\n      "end_char": 18\n    },\n    {\n      "id": 5,\n      "text": ".",\n      "lemma": ".",\n      "upos": "PUNCT",\n      "xpos": ".",\n      "start_char": 18,\n      "end_char": 19\n    }\n  ],\n  [\n    {\n      "id": 1,\n      "text": "And",\n      "lemma": "and",\n      "upos": "CCONJ",\n      "xpos": "CC",\n      "start_char": 20,\n      "end_char": 23\n    },\n    {\n      "id": 2,\n      "text": "here",\n      "lemma": "here",\n      "upos": "ADV",\n      "xpos": "RB",\n      "feats": "PronType=Dem",\n      "start_char": 24,\n      "end_char": 28\n    },\n    {\n      "id": 3,\n      "text": "we",\n      "lemma": "we",\n      "upos": "PRON",\n      "xpos": "PRP",\n      "feats": "Case=Nom|Number=Plur|Person=1|PronType=Prs",\n      "start_char": 29,\n      "end_char": 31\n    },\n    {\n      "id": 4,\n      "text": "go",\n      "lemma": "go",\n      "upos": "VERB",\n      "xpos": "VBP",\n      "feats": "Mood=Ind|Number=Plur|Person=1|Tense=Pres|VerbForm=Fin",\n      "start_char": 32,\n      "end_char": 34\n    },\n    {\n      "id": 5,\n      "text": ".",\n      "lemma": ".",\n      "upos": "PUNCT",\n      "xpos": ".",\n      "start_char": 34,\n      "end_char": 35\n    }\n  ]\n]"""


def test_fix_dict_path():
    mydict = {
        "dir": "/home/inga/stanza_resources",
        "tokenize_model_path": "en/tokenize/combined.pt",
    }
    mydict = ma.mstanza_preprocess.fix_dict_path(mydict)
    test_mydict = {
        "dir": "/home/inga/stanza_resources",
        "tokenize_model_path": "/home/inga/stanza_resources/en/tokenize/combined.pt",
    }
    assert mydict == test_mydict


def test_init_pipeline():
    mydict = {
        "lang": "en",
        "dir": "/home/inga/stanza_resources",
        "processors": "tokenize,pos,lemma",
    }
    obj = ma.mstanza_pipeline(mydict)
    obj.init_pipeline()


def test_process_text():
    mydict = {
        "lang": "en",
        "dir": "/home/inga/stanza_resources",
        "processors": "tokenize,pos,lemma",
    }
    text = "This is an example. And here we go."
    obj = ma.mstanza_pipeline(mydict)
    obj.init_pipeline()
    doc = obj.process_text(text)
    assert str(doc) == str(test_doc)


# output object
# def test_assemble_output_sent():
#     data = "This is an example. And here we go."
#     nlp = sp.load("en_core_web_md")
#     doc = nlp(data)
#     jobs = ["tok2vec", "senter", "tagger", "parser",
#     "attribute_ruler", "lemmatizer", "ner"]
#     start = 0
#     out = be.out_object.assemble_output_sent(doc, jobs, start)
#     print(out)
