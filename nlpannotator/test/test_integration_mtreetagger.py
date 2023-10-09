import nlpannotator.base as be
import nlpannotator.mtreetagger as mtt
import pytest
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


@pytest.mark.treetagger
def test_integration_mtreetagger(tmp_path):
    data = "This is a sentence."
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    mydict["tool"] = "treetagger"
    mydict["treetagger_dict"]["processors"] = "tokenize", "pos", "lemma"
    inputfile = pkg / "data" / "example_en.txt"
    mydict["input"] = inputfile.as_posix()
    outdir = tmp_path / "out"
    corpusdir = tmp_path / "corpora"
    registrydir = tmp_path / "registry"
    mydict["advanced_options"]["output_dir"] = outdir.as_posix()
    mydict["advanced_options"]["corpus_dir"] = corpusdir.as_posix()
    mydict["advanced_options"]["registry_dir"] = registrydir.as_posix()
    treetagger_dict = mydict["treetagger_dict"]
    annotated = mtt.MyTreetagger(treetagger_dict)
    annotated = annotated.apply_to(data)
    start = 0
    out_obj = mtt.OutTreetagger(annotated.doc, annotated.jobs, start=start)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.ptags
    stags = ["s"]
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    encode_obj = be.EncodeCorpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
