import base as be
import pipe as pe
import mspacy as msp


if __name__ == "__main__":
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    mydict["processing_option"] = "accurate"
    mydict["processing_type"] = "sentencize, pos  ,lemma, tokenize"
    mydict["advanced_options"]["output_dir"] = "./src/annotator/test/out/"
    mydict["advanced_options"]["corpus_dir"] = "./src/annotator/test/corpora/"
    mydict["advanced_options"]["registry_dir"] = "./src/annotator/test/registry/"
    be.prepare_run.validate_input_dict(mydict)
    pe.SetConfig(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    data = be.prepare_run.get_text("./src/annotator/test/test_files/example_de.txt")
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    # annotated.pass_results(mydict)
    # instead call the methods from here
    start = 0
    out_obj = msp.out_object_spacy(annotated.doc, annotated.jobs, start=start)
    style = "STR"
    out = out_obj.fetch_output(style)
    ptags = None
    ptags = ptags or out_obj.ptags
    stags = out_obj.stags
    # write to file -> This overwrites any existing file of given name;
    # as all of this should be handled internally and the files are only
    # temporary, this should not be a problem. right?
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    add = False
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
    # be.out_object.write_xml(
    # mydict["output"].replace("/", "_"), mydict["output"], out
    # )

    # elif ret is True:
    # return out
    # else:
    # raise RuntimeError(
    # "If ret is not set to True, a dict containing the encoding parameters is needed!"
    # )
