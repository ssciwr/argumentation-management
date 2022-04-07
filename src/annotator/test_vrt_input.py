import annotator.base as be
import annotator.mspacy as msp
import annotator.mstanza as ma
import annotator.pipe as pe

if __name__ == "__main__":

    data = [
        ["This", "is", "an", "example", "text", "."],
        ["This", "is", "a", "second", "sentence", "."],
        ["It", "has-", "some", "weird.", "tokenization!"],
    ]

    data_untokenized = ""
    for lists in data:
        for string in lists:
            data_untokenized += string + " "
    # data = be.prepare_run.load_tokens_from_vrt("./test/test_files/example_en_spacy.vrt")
    # print(data)

    mydict = be.prepare_run.load_input_dict("./input")
    mydict["tool"] = "spacy"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    be.prepare_run.validate_input_dict(mydict)

    spacy_dict = mydict["spacy_dict"]

    pipe = msp.spacy_pipe(spacy_dict, pretokenized=True)

    annotated = pipe.apply_to(data)

    out = annotated.pass_results("STR", mydict, ret=True)

    print(out)
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

    try:
        assert out == check

    except AssertionError:
        print("unequal")

    data = be.prepare_run.load_tokens_from_vrt("./test/test_files/example_en_test.vrt")

    mydict = be.prepare_run.load_input_dict("./input")
    mydict = pe.SetConfig.set_processors(mydict)
    mydict["tool"] = "stanza"
    mydict["input"] = "./test/test_files/example_de.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    # validate the input dict
    be.prepare_run.validate_input_dict(mydict)
    stanza_dict = mydict["stanza_dict"]
    stanza_dict["lang"] = "en"
    stanza_dict["processors"] = "tokenize,pos,mwt,lemma"
    stanza_dict["dir"] = "./test/models/"
    # stanza_dict = be.prepare_run.update_dict(stanza_dict)
    stanza_dict = be.prepare_run.activate_procs(stanza_dict, "stanza_")
    # initialize the pipeline with the dict
    stanza_pipe = ma.MyStanza(stanza_dict, pretokenized=True)

    out = stanza_pipe.apply_to(data).pass_results(mydict, ret=True)

    organised_out = []
    for item in out:
        organised_out.append(item.strip("\n").split("\t"))

    print(organised_out)
