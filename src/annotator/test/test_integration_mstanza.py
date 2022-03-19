import base as be
import mstanza as ma


def test_integration_mstanza():
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["tool"] = "stanza"
    mydict["input"] = "./test/test_files/example_de.txt"
    mydict["advanced_options"]["output_dir"] = "../../out/"
    stanza_dict = mydict["stanza_dict"]
    stanza_dict["lang"] = "de"
    stanza_dict["processors"] = "tokenize,pos,mwt,lemma"
    stanza_dict["dir"] = "./test/models/"
    stanza_dict = be.prepare_run.update_dict(stanza_dict)
    stanza_dict = be.prepare_run.activate_procs(stanza_dict, "stanza_")
    data = be.prepare_run.get_text(mydict["input"])
    # initialize the pipeline with the dict
    stanza_pipe = ma.Stanza(stanza_dict)
    # apply pipeline to data and encode
    stanza_pipe.apply_to(data).pass_results(mydict)
    # maybe here assert that written vrt is same as safe version
