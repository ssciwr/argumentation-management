import base as be
import mspacy as msp


def test_integration_mspacy():
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./input")
    mydict["tool"] = "spacy"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"
    be.prepare_run.validate_input_dict(mydict)
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = msp.spacy_pipe(spacy_dict)
    data = be.prepare_run.get_text(mydict["input"])
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results("STR", mydict, ret=False)
