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
    annotated.pass_results(mydict)
