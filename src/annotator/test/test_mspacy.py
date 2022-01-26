import pytest
import spacy as sp
import mspacy as msp
import base as be


def test_init():
    """Check if the parameters from the input dict are loaded into the
    pipe object as expected."""

    # assuming that base functions are already tested in base
    # mydict = be.prepare_run.load_input_dict("test/input_short")
    # if we use "outname", pass the full dict
    mydict = be.prepare_run.load_input_dict("test/input")
    mydict_test = be.prepare_run.load_input_dict("test/input_short")

    test_obj = msp.spacy_pipe(mydict)

    assert test_obj.outname == mydict["output"]
    assert test_obj.pretrained == mydict_test["pretrained"]
    assert test_obj.lang == mydict_test["lang"]
    assert test_obj.type == mydict_test["text_type"]
    assert test_obj.model == mydict_test["model"]
    assert test_obj.jobs == [
        proc.strip() for proc in mydict_test["processors"].split(",")
    ]
    assert test_obj.config == be.prepare_run.update_dict(mydict_test["config"])

    return test_obj, mydict_test


def test_pipe_sent():
    """Check if applying pipeline through mspacy leads to same result as applying
    same pipeline through spacy directly."""

    text = """This is an example text. This is a second sentence."""

    test_obj, mydict = test_init()

    # as this pipe config should just load all of a model, results
    # should be equivalent to using:
    nlp = sp.load(mydict["model"])
    check_doc = nlp(text)

    test_doc = test_obj.apply_to(text).doc

    assert test_doc.has_annotation("SENT_START")
    # there seems to be no __eq__() definition for either doc or token in spacy
    # so just compare a couple of attributes for every token?
    for test_token, token in zip(test_doc, check_doc):

        assert test_token.text == token.text
        assert test_token.lemma == token.lemma
        assert test_token.pos == token.pos
        assert test_token.tag == token.tag
        assert test_token.sent_start == token.sent_start

    return test_obj.apply_to(text), check_doc


def test_output_sent():
    """Check if output is as expected, use current output as example result.
    Additionally use doc build through spacy directly and compare output."""

    # this is quite specific, any way to generalize?
    test_obj, check_doc = test_pipe_sent()

    test_out = msp.out_object_spacy(test_obj.doc, test_obj.jobs, start=0).fetch_output()

    check_out = msp.out_object_spacy(check_doc, test_obj.jobs, start=0).fetch_output()

    # as the below two are different, is the outcome of the analysis platform-dependant?
    # check = ['<s>\n', 'This\tDT\tthis\t-\tnsubj\tPRON\n', 'is\tVBZ\tbe\t-\tROOT\tAUX\n', 'an\tDT\tan\t-\tdet\tDET\n', 'example\tNN\texample\t-\tcompound\tNOUN\n', 'text\tNN\ttext\t-\tattr\tNOUN\n', '.\t.\t.\t-\tpunct\tPUNCT\n', '</s>\n', '<s>\n', 'This\tDT\tthis\t-\tnsubj\tPRON\n', 'is\tVBZ\tbe\t-\tROOT\tAUX\n', 'a\tDT\ta\t-\tdet\tDET\n', 'second\tJJ\tsecond\tORDINAL\tamod\tADJ\n', 'sentence\tNN\tsentence\t-\tattr\tNOUN\n', '.\t.\t.\t-\tpunct\tPUNCT\n', '</s>\n']
    check = [
        "<s>\n",
        "This\tDT\tthis\t-\tnsubj\tPRON\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "an\tDT\tan\t-\tdet\tDET\n",
        "example\tNN\texample\t-\tcompound\tNOUN\n",
        "text\tNN\ttext\t-\tattr\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
        "<s>\n",
        "This\tDT\tthis\t-\tnsubj\tPRON\n",
        "is\tVBZ\tbe\t-\tROOT\tAUX\n",
        "a\tDT\ta\t-\tdet\tDET\n",
        "second\tJJ\tsecond\tORDINAL\tamod\tADJ\n",
        "sentence\tNN\tsentence\t-\tattr\tNOUN\n",
        ".\t.\t.\t-\tpunct\tPUNCT\n",
        "</s>\n",
    ]
    assert test_out == check_out
    assert test_out == check


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
if __name__ == "__main__":
    test_init()
    test_pipe_sent()
    test_output_sent()
