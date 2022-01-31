import pytest
import mstanza as ma

mydict_en = {"lang": "en", "dir": "./test/models/", "processors": "tokenize,pos,lemma"}
mydict_de = {"lang": "en", "dir": "./test/models/", "processors": "tokenize,pos,lemma"}


@pytest.fixture()
def set_file_dict():
    text_dict = {
        "en": "./test/example_en.txt",
        "de": "./test/example_de.txt",
        "test_en": "./test/example_en_stanza.txt",
        "test_de": "./test/example_de_stanza.txt",
        "tok": "./test/example_en_stanza_tok.txt",
        "tok_pos": "./test/example_en_stanza_tok_pos.txt",
        "tok_pos_lemma": "./test/example_en_stanza_tok_pos_lemma.txt",
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
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()


@pytest.mark.lang("en")
def test_process_text_en(get_sample, get_sample_stanza):
    text = get_sample
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # clean up the returned object, convert to string
    doc = str(docobj).replace("\n", "")
    doc = doc.replace(" ", "")
    test_doc = get_sample_stanza.replace(" ", "")
    assert doc == test_doc


@pytest.mark.lang("de")
def test_process_text_de(get_sample, get_sample_stanza):
    text = get_sample
    obj = ma.mstanza_pipeline(mydict_de)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # clean up the returned object, convert to string
    doc = str(docobj).replace("\n", "")
    doc = doc.replace(" ", "")
    test_doc = get_sample_stanza.replace(" ", "")
    assert doc == test_doc


@pytest.mark.lang("en")
@pytest.mark.proc("tok")
def test_out_object_stanza_tok(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize"
    mydict_en["processors"] = procstring
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # now call the postprocessing
    out = ma.out_object_stanza.assemble_output_sent(docobj, procstring, start=0)
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
@pytest.mark.proc("tok_pos")
def test_out_object_stanza_tok_pos(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize,pos"
    mydict_en["processors"] = procstring
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # now call the postprocessing
    out = ma.out_object_stanza.assemble_output_sent(docobj, procstring, start=0)
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
@pytest.mark.proc("tok_pos_lemma")
def test_out_object_stanza_tok_pos_lemma(get_sample, get_out_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    mydict_en["processors"] = procstring
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # now call the postprocessing
    out = ma.out_object_stanza.assemble_output_sent(docobj, procstring, start=0)
    test_out = get_out_sample
    # compare as string not as list
    # reading in as list will add further \n
    assert str(out) == test_out


@pytest.mark.lang("en")
def test_out_object_stanza_vrt(get_sample):
    text = get_sample
    procstring = "tokenize,pos,lemma"
    outfile = "./test/example_en"
    obj = ma.mstanza_pipeline(mydict_en)
    obj.init_pipeline()
    docobj = obj.process_text(text)
    # now call the postprocessing
    out = ma.out_object_stanza.assemble_output_sent(docobj, procstring, start=0)
    file_out = open(outfile + "_test.vrt", "r")
    # call vrt writing
    ma.out_object_stanza.write_vrt(outfile, out)
    file = open(outfile + ".vrt", "r")
    assert file.read() == file_out.read()
    import os

    if os.path.exists(outfile + ".vrt"):
        os.remove(outfile + ".vrt")
    else:
        print("{} was not created during testing.".format(outfile + ".vrt"))
