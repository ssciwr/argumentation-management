import pytest
import mstanza as ma


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
