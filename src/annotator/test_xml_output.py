import annotator.base as be
import annotator.mspacy as ms

if __name__ == "__main__":
    data = be.prepare_run.get_sample_text()
    # or read the main dict and activate
    mydict = be.prepare_run.load_input_dict("src/annotator/input")
    # take only the part of dict pertaining to spacy
    # filename needs to be moved to/taken from top level of dict
    # spacy_dict = mydict["spacy_dict"]
    # remove comment lines starting with "_"
    # for now, we are not using "components" as these are defined through the pre-
    # made models; for making your own model, they will need to be used
    # we will worry about this later
    spacy_dict = be.prepare_run.update_dict(mydict)
    # build pipe from config, apply it to data, write results to vrt
    # spacy_pipe(spacy_dict).pply_to(data).pass_results()
    # if we use "outname", this needs to be passed the full dict
    ms.spacy_pipe(mydict).apply_to(data).pass_results("DICT")
