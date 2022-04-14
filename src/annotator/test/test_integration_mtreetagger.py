import pytest
import annotator.base as be
import annotator.mtreetagger as mtt


def test_integration_mtreetagger():
    data = be.prepare_run.get_text("test/test_files/example_en.txt")
    mydict = be.prepare_run.load_input_dict("input")
    mydict["tool"] = "treetagger"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "./test/test_files/"

    treetagger_dict = mydict["treetagger_dict"]
    pipe = mtt.treetagger_pipe(treetagger_dict)

    out = pipe.apply_to(data).pass_results(mydict)
