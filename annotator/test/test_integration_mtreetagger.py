import pytest

# from .context import base as be
import base as be

# from .context import mtreetagger as mtt
import mtreetagger as mtt
from tempfile import TemporaryDirectory


def test_integration_mtreetagger():
    out = TemporaryDirectory()

    data = be.prepare_run.get_text("test/test_files/example_en.txt")
    mydict = be.prepare_run.load_input_dict("input")
    mydict["tool"] = "treetagger"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)

    treetagger_dict = mydict["treetagger_dict"]
    pipe = mtt.treetagger_pipe(treetagger_dict)

    out = pipe.apply_to(data).pass_results(mydict)
