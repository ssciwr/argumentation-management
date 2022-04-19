import annotator.base as be
import annotator.mtreetagger as mtt
import annotator.mspacy as msp

if __name__ == "__main__":
    data = be.prepare_run.get_text("src/annotator/test/test_files/example_en.txt")
    mydict = be.prepare_run.load_input_dict("src/annotator/input_local")

    treetagger_dict = mydict["treetagger_dict"]
    pipe = mtt.treetagger_pipe(treetagger_dict)

    out = pipe.apply_to(data)

    out.pass_results(mydict, "STR")

    mydict["tool"] = "spacy"
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results(
        mydict, add=True, ptags=["test1", "test2", "test3", "test4", "test5"]
    )
    be.decode_corpus(mydict).decode_to_file("out/")
