import annotator.base as be
import annotator.mspacy as ms
import annotator.mstanza as mst

if __name__ == "__main__":
    data = be.prepare_run.get_sample_text()
    # or read the main dict and activate
    mydict = be.prepare_run.load_input_dict("src/annotator/input_local")
    # take only the part of dict pertaining to spacy
    # filename needs to be moved to/taken from top level of dict
    # spacy_dict = mydict["spacy_dict"]
    # remove comment lines starting with "_"
    # for now, we are not using "components" as these are defined through the pre-
    # made models; for making your own model, they will need to be used
    # we will worry about this later

    # spacy

    # spacy_dict = mydict["spacy_pipe"]
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # ms.spacy_pipe(spacy_dict).apply_to(data).pass_results("STR", encoding_dict)

    # stanza

    stanza_dict = mydict["stanza_dict"]

    stanza_pipe = mst.mstanza_pipeline(stanza_dict)
    stanza_pipe.init_pipeline()

    results = stanza_pipe.process_text(data)

    encoding_dict = be.prepare_run.get_encoding(mydict)

    stanza_pipe.postprocess(encoding_dict)

    be.decode_corpus(encoding_dict).decode_to_file(directory="out")
