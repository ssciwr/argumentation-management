import os
import pytest
import nlpannotator.base as be
import nlpannotator.mstanza as ma

mydict_en = be.PrepareRun.load_input_dict("./test/data/test_stanza_en.json")[
    "stanza_dict"
]
mydict_de = be.PrepareRun.load_input_dict("./test/data/test_stanza_de.json")[
    "stanza_dict"
]

mydict_en = {"lang": "en", "dir": "./test/models/", "processors": "tokenize,pos,lemma"}
mydict_de = {"lang": "de", "dir": "./test/models/", "processors": "tokenize,pos,lemma"}


@pytest.fixture()
def set_file_dict():
    text_dict = {
        "en": "./test/data/example_en.txt",
        "de": "./test/data/example_de.txt",
        "test_en": "./test/data/example_en_stanza.txt",
        "test_de": "./test/data/example_de_stanza.txt",
        "tok": "./test/data/example_en_stanza_tok.txt",
        "tok_pos": "./test/data/example_en_stanza_tok_pos.txt",
        "tok_pos_lemma": "./test/data/example_en_stanza_tok_pos_lemma.txt",
    }
    return text_dict


@pytest.fixture()
def get_sample(set_file_dict, request):
    """Load the specified sample text."""
    marker = request.node.get_closest_marker("lang")
    if marker == "None":
        # missing marker
        print("Missing a marker for reading the sample text.")
    else:
        lang_key = marker.args[0]
    # Read the sample text.
    with open(set_file_dict[lang_key], "r") as myfile:
        data = myfile.read().replace("\n", "")
    return data


@pytest.fixture()
def get_sample_stanza(set_file_dict, request):
    """Load the specified sample text output from stanza."""
    marker = request.node.get_closest_marker("lang")
    if marker == "None":
        # missing marker
        print("Missing a marker for reading the sample text output from stanza.")
    else:
        lang_key = marker.args[0]
    # Read the processed text from stanza.
    with open(set_file_dict["test_" + lang_key], "r") as myfile:
        test_data = myfile.read().replace("\n", "")
    return test_data


@pytest.fixture()
def get_out_sample(set_file_dict, request):
    """Load the specified vrt sample."""
    marker = request.node.get_closest_marker("proc")
    if marker == "None":
        # missing marker
        print("Missing a marker for reading the vrt sample text.")
    else:
        proc_key = marker.args[0]
    # Read the sample text.
    with open(set_file_dict[proc_key], "r") as myfile:
        data = myfile.read()
    return data


@pytest.mark.lang("en")
def test_apply_to_en(get_sample, get_sample_stanza):
    text = get_sample
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # clean up the returned object, convert to string
    doc = str(docobj.doc).replace("\n", "")
    doc = doc.replace(" ", "")
    test_doc = get_sample_stanza
    assert doc == test_doc


@pytest.mark.lang("de")
def test_apply_to_de(get_sample, get_sample_stanza):
    text = get_sample
    obj = ma.MyStanza(mydict_de)
    docobj = obj.apply_to(text)
    # clean up the returned object, convert to string
    doc = str(docobj.doc).replace("\n", "")
    doc = doc.replace(" ", "")
    test_doc = get_sample_stanza.replace(" ", "")
    assert doc == test_doc


@pytest.mark.lang("en")
@pytest.mark.proc("tok")
def test_outstanza_tok(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize"
    mydict_en["processors"] = procstring
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # now call the postprocessing
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    out = out_obj.assemble_output_sent()
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
@pytest.mark.proc("tok_pos")
def test_outstanza_tok_pos(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize,pos"
    mydict_en["processors"] = procstring
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # now call the postprocessing
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
@pytest.mark.proc("tok_pos_lemma")
def test_outstanza_tok_pos_lemma(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    mydict_en["processors"] = procstring
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # now call the postprocessing
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
def test_outstanza_vrt(get_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    outfile = "./test/data/example_en"
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # now call the postprocessing
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    file_out = open(outfile + "_test.vrt", "r")
    # call vrt writing
    ma.OutStanza.write_vrt(outfile, out)
    file = open(outfile + ".vrt", "r")
    print("wrote " + outfile + ".vrt")
    assert file.read() == file_out.read()
    if os.path.exists(outfile + ".vrt"):
        os.remove(outfile + ".vrt")
    else:
        print("{} was not created during testing.".format(outfile + ".vrt"))


@pytest.mark.lang("en")
def test_token_list(get_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    token_list = []
    for sent in docobj.doc.sentences:
        token_list += out_obj.token_list(sent)

    with open("./test/data/example_en_token_list.txt", "r") as file:
        out_token_list = file.read()

    assert str(token_list) == out_token_list


@pytest.mark.lang("en")
def test_word_list(get_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    word_list = []
    for sent in docobj.doc.sentences:
        word_list += out_obj.word_list(sent)

    with open("./test/data/example_en_word_list.txt", "r") as file:
        out_word_list = file.read()

    assert str(word_list) == out_word_list


@pytest.mark.lang("en")
def test_sentences(get_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    sentences = out_obj.sentences
    with open("./test/data/example_en_sentences.txt", "r") as file:
        out_sentences = file.read()

    assert str(sentences) == out_sentences
