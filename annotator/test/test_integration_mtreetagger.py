import base as be
import mtreetagger as mtt
from tempfile import TemporaryDirectory


def test_integration_mtreetagger():
    out = TemporaryDirectory()
    data = "This is a sentence."
    mydict = be.PrepareRun.load_input_dict("input")
    mydict["tool"] = "treetagger"
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    mydict["input"] = "./test/test_files/example_en.txt"
    mydict["advanced_options"]["output_dir"] = "{}".format(out.name)

    treetagger_dict = mydict["treetagger_dict"]
    annotated = mtt.MyTreetagger(treetagger_dict)
    annotated = annotated.apply_to(data)
    start = 0
    out_obj = mtt.OutTreetagger(annotated.doc, annotated.jobs, start=start)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.ptags
    stags = out_obj.get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
