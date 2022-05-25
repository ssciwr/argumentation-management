import base as be
import pipe as pe
import mspacy as msp


def call_spacy(mydict, data):
    # lets start with setting it up for spacy
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline
    pipe = msp.spacy_pipe(spacy_dict)
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # we should not need start ..?
    start = 0
    out_obj = msp.OutSpacy(annotated.doc, annotated.jobs, start=start)
    # should return sentences as well in case of senter
    # TODO: set up the processing options corretly- senter, parser
    return out_obj


def call_stanza(mydict, data):
    pass


switch_tool = {"spacy": call_spacy, "stanza": call_stanza}

if __name__ == "__main__":
    # load input dict
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    # overwrite defaults for testing purposes
    mydict["processing_option"] = "accurate"
    mydict["processing_type"] = "sentencize, pos  ,lemma, tokenize"
    mydict["advanced_options"]["output_dir"] = "./src/annotator/test/out/"
    mydict["advanced_options"]["corpus_dir"] = "./src/annotator/test/corpora/"
    mydict["advanced_options"]["registry_dir"] = "./src/annotator/test/registry/"
    # get the data to be processed
    data = be.prepare_run.get_text("./src/annotator/test/test_files/example_de.txt")
    # validate the input dict
    be.prepare_run.validate_input_dict(mydict)
    # activate the input dict
    pe.SetConfig(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
    print(mydict["processing_option"], "option")
    print(mydict["tool"], "tool")
    print(set(mydict["tool"]), "tool")
    for mytool in set(mydict["tool"]):
        print(mytool)
        # Now call specific routines
        out_obj = switch_tool[mytool](mydict, data)

    # the below for generating the output
    # for xml or vrt, let's stick with vrt for now - TODO
    style = "STR"
    out = out_obj.fetch_output(style)
    # find out if there are user-defined tags
    ptags = None
    ptags = ptags or out_obj.ptags
    stags = out_obj.stags
    # write to file -> This overwrites any existing file of given name;
    # as all of this should be handled internally and the files are only
    # temporary, this should not be a problem. right?
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    # to use pretokenized data - TODO
    # add = False
    # not sure why we need so many cases - TODO
    # if ret is False and style == "STR" and mydict is not None and add is False:
    # if not add:
    out_obj.write_vrt(outfile, out)
    # encode
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
    # elif ret is False and style == "STR" and mydict is not None and add is True:
    # else:
    #   out_obj.write_vrt(outfile, out)
    #   encode_obj = be.encode_corpus(mydict)
    #   encode_obj.encode_vrt(ptags, stags)
    # elif ret is False and style == "DICT" and mydict is not None:
    # be.OutObject.write_xml(
    # mydict["output"].replace("/", "_"), mydict["output"], out
    # )
    # elif ret is True:
    # return out
    # else:
    # raise RuntimeError(
    # "If ret is not set to True, a dict containing the encoding parameters is needed!"
    # )
