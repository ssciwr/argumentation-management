import base as be
import pipe as pe
import mspacy as msp
import mstanza as msa


def call_spacy(mydict, data, islist=False):
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline
    annotated = msp.MySpacy(spacy_dict)
    # apply pipeline to data
    if not islist:
        annotated.apply_to(data)
        doc = annotated.doc
    else:
        doc = []
        for sentence in data:
            annotated.apply_to(sentence)
            doc.append(annotated.doc)
    # we should not need start ..?
    start = 0
    out_obj = msp.OutSpacy(doc, annotated.jobs, start=start, islist=islist)
    return out_obj


def call_stanza(mydict, data, islist=False):
    stanza_dict = mydict["stanza_dict"]
    # load the pipeline
    annotated = msa.MyStanza(stanza_dict)
    # apply pipeline to data
    if not islist:
        annotated.apply_to(data)
        doc = annotated.doc
    else:
        doc = []
        for sentence in data:
            annotated.apply_to(sentence)
            doc.append(annotated.doc)
    # we should not need start ..?
    start = 0
    out_obj = msa.out_object_stanza(
        annotated.doc, annotated.jobs, start=start, islist=islist
    )
    return out_obj


call_tool = {"spacy": call_spacy, "stanza": call_stanza}

if __name__ == "__main__":
    # load input dict
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    # overwrite defaults for testing purposes
    mydict["processing_option"] = "accurate"
    # mydict["processing_option"] = "fast"
    mydict["processing_type"] = "sentencize, pos  ,lemma, tokenize"
    mydict["language"] = "en"
    # mydict["language"] = "de"
    mydict["advanced_options"]["output_dir"] = "./src/annotator/test/out/"
    mydict["advanced_options"]["corpus_dir"] = "./src/annotator/test/corpora/"
    mydict["advanced_options"]["registry_dir"] = "./src/annotator/test/registry/"
    # get the data to be processed
    data = be.prepare_run.get_text("./src/annotator/test/test_files/example_en.txt")
    # data = be.prepare_run.get_text("./src/annotator/test/test_files/example_de.txt")
    # validate the input dict
    be.prepare_run.validate_input_dict(mydict)
    # activate the input dict
    pe.SetConfig(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
    print(mydict["processing_option"], "option")
    print(mydict["tool"], "tool")
    print(set(mydict["tool"]), "tool")
    out_obj = []
    data_islist = False
    for mytool in set(mydict["tool"]):
        print(mytool)
        # if sentences are data, then we need to go through list
        # call specific routines
        out_obj.append(call_tool[mytool](mydict, data, data_islist))
        # the first tool will sentencize
        # all subsequent ones will use sentencized input
        # so the new data is sentences from first tool
        # however, this is now a list
        data = out_obj[0].sentences
        data_islist = True
    # stanza
    # the below for generating the output
    # for xml or vrt, let's stick with vrt for now - TODO
    # style = "STR"
    out = out_obj[0].assemble_output_sent()
    # spacy
    # this replicates functionality, we need assemble_output_sent instead
    # out = out_obj[0].fetch_output(style)
    ptags = None
    ptags = ptags or out_obj[0].get_ptags()
    stags = out_obj[0].get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj[0].write_vrt(outfile, out)
    # if not add:
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
    # elif add:
    #     encode_obj = be.encode_corpus(mydict)
    #     encode_obj.add_tags_to_corpus(mydict, ptags, stags)

    # to use pretokenized data - TODO
    # not sure why we need so many cases - TODO
    # if ret is False and style == "STR" and mydict is not None and add is False:
    # if not add:
    # elif ret is False and style == "STR" and mydict is not None and add is True:
    # elif ret is True:
    # "If ret is not set to True, a dict containing the encoding parameters is needed!"
