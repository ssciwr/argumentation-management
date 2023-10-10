import os
import pytest
import nlpannotator.base as be
import nlpannotator.mstanza as ma
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")

mydict_en = be.PrepareRun.load_input_dict(pkg / "data" / "test_stanza_en.json")[
    "stanza_dict"
]
mydict_de = be.PrepareRun.load_input_dict(pkg / "data" / "test_stanza_de.json")[
    "stanza_dict"
]

mydict_en = {
    "lang": "en",
    "dir": "stanza_resources",
    "processors": "tokenize,pos,lemma",
}
mydict_de = {
    "lang": "de",
    "dir": "stanza_resources",
    "processors": "tokenize,pos,lemma",
}


@pytest.fixture()
def set_file_dict():
    text_dict = {
        "en": (pkg / "data" / "example_en.txt").as_posix(),
        "de": (pkg / "data" / "example_de.txt").as_posix(),
        "test_en": (pkg / "data" / "example_en_stanza.txt").as_posix(),
        "test_de": (pkg / "data" / "example_de_stanza.txt").as_posix(),
        "tok": (pkg / "data" / "example_en_stanza_tok.txt").as_posix(),
        "tok_pos": (pkg / "data" / "example_en_stanza_tok_pos.txt").as_posix(),
        "tok_pos_lemma": (
            pkg / "data" / "example_en_stanza_tok_pos_lemma.txt"
        ).as_posix(),
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


@pytest.mark.treetagger
@pytest.mark.lang("de")
def test_apply_to_de(get_sample, get_sample_stanza):
    text = get_sample
    obj = ma.MyStanza(mydict_de)
    docobj = obj.apply_to(text)
    # clean up the returned object, convert to string
    doc = str(docobj.doc).replace("\n", "")
    doc = doc.replace(" ", "")
    test_doc = get_sample_stanza.replace(" ", "")
    assert doc[0:100] == test_doc[0:100]


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
    outfile = pkg / "data" / "example_en"
    obj = ma.MyStanza(mydict_en)
    docobj = obj.apply_to(text)
    # now call the postprocessing
    out_obj = ma.OutStanza(docobj.doc, procstring, start=0)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    file_out = open(outfile.as_posix() + "_test.vrt", "r")
    # call vrt writing
    ma.OutStanza.write_vrt(outfile.as_posix(), out)
    file = open(outfile.as_posix() + ".vrt", "r")
    print("wrote " + outfile.as_posix() + ".vrt")
    assert file.read() == file_out.read()
    file.close()
    if os.path.exists(outfile.as_posix() + ".vrt"):
        os.remove(outfile.as_posix() + ".vrt")
    else:
        print("{} was not created during testing.".format(outfile.as_posix() + ".vrt"))


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
    inputfile = pkg / "data" / "example_en_token_list.txt"
    with open(inputfile.as_posix(), "r") as file:
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
    inputfile = pkg / "data" / "example_en_word_list.txt"
    with open(inputfile.as_posix(), "r") as file:
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
    inputfile = pkg / "data" / "example_en_sentences.txt"
    with open(inputfile.as_posix(), "r") as file:
        out_sentences = file.read()

    assert str(sentences) == out_sentences
