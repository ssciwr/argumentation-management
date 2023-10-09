import pytest
import nlpannotator.base as be
import nlpannotator.pipe as pe
import nlpannotator.mstanza as ma
import importlib_resources

pkg = importlib_resources.files("nlpannotator.test")


def test_integration_mstanza(tmp_path):
    # read in input.json
    mydict = be.PrepareRun.load_input_dict(pkg / "data" / "input.json")
    inputfile = pkg / "data" / "example_de.txt"
    mydict["input"] = inputfile.as_posix()
    mydict["tool"] = "stanza, stanza, stanza, stanza"
    mydict["language"] = "de"
    mydict["document_type"] = "text"
    mydict["processing_option"] = "manual"
    mydict["processing_type"] = "sentencize,tokenize,pos,lemma"
    outdir = tmp_path / "out"
    corpusdir = tmp_path / "corpora"
    registrydir = tmp_path / "registry"
    mydict["advanced_options"]["output_dir"] = outdir.as_posix()
    mydict["advanced_options"]["corpus_dir"] = corpusdir.as_posix()
    mydict["advanced_options"]["registry_dir"] = registrydir.as_posix()
    # validate the input dict
    be.PrepareRun.validate_input_dict(mydict)
    # load the pipe object for updating dict with settings
    obj = pe.SetConfig(mydict)
    stanza_dict = obj.mydict["stanza_dict"]
    stanza_dir = pkg / "models"
    stanza_dict["dir"] = stanza_dir.as_posix()
    data = be.PrepareRun.get_text(mydict["input"])
    # initialize the pipeline with the dict
    stanza_pipe = ma.MyStanza(stanza_dict)
    # apply pipeline to data
    stanza_pipe.apply_to(data)
    start = 0
    out_obj = ma.OutStanza(stanza_pipe.doc, stanza_pipe.jobs, start=start)
    out = out_obj.assemble_output_sent()
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.ptags
    stags = out_obj.get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    encode_obj = be.EncodeCorpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
