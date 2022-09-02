import nlpannotator.base as be
import nlpannotator.pipe as pe
import nlpannotator.mspacy as msp
import nlpannotator.mstanza as msa
import nlpannotator.msomajo as mso
import nlpannotator.mtreetagger as mtt
import nlpannotator.mflair as mf


def call_spacy(mydict, data, islist=False, style="STR"):
    spacy_dict = mydict["spacy_dict"]
    # load the pipeline
    annotated = msp.MySpacy(spacy_dict)
    # apply pipeline to data
    # data is not a list of sentences and will generate one doc object
    if not islist:
        annotated.apply_to(data)
        doc = annotated.doc
    else:
        # data is a list of sentences and will generate a list of doc objects
        doc = []
        for sentence in data:
            annotated.apply_to(sentence)
            doc.append(annotated.doc)
    # we should not need start ..?
    start = 0
    out_obj = msp.OutSpacy(doc, annotated.jobs, start=start, style=style)
    return out_obj


def call_stanza(mydict, data, islist=False, style="STR"):
    stanza_dict = mydict["stanza_dict"]
    if islist:
        stanza_dict["tokenize_no_ssplit"] = True
        # stanza needs tokenizer or tokenize_pretokenized=True
        # we want to avoid the latter so set the tokenizer
        # in some cases it could happen that tokenization differs from the tools
        # but we will walk that path when we get there
        stanza_dict["processors"] = "tokenize," + stanza_dict["processors"]
        # https://stanfordnlp.github.io/stanza/tokenize.html#start-with-pretokenized-text
        # set two continuous newlines so that sentences are not
        # split but we still use efficient capabilities
        data = [sent + "\n\n" for sent in data]
    # load the pipeline
    annotated = msa.MyStanza(stanza_dict)
    # apply pipeline to data
    annotated.apply_to(data)
    doc = annotated.doc
    # we should not need start ..?
    start = 0
    out_obj = msa.OutStanza(doc, annotated.jobs, start=start, style=style)
    return out_obj


def call_somajo(mydict, data, islist=False, style="STR"):
    somajo_dict = mydict["somajo_dict"]
    # load the pipeline
    # somajo does only sentence-split and tokenization
    tokenized = mso.MySomajo(somajo_dict)
    # apply pipeline to data
    tokenized.apply_to(data)
    # we should not need start ..?
    start = 0
    # for somajo we never have list data as this will be only used for sentencizing
    out_obj = mso.OutSomajo(tokenized.doc, tokenized.jobs, start=start, style=style)
    return out_obj


def call_treetagger(mydict, data, islist=True, style="STR"):
    treetagger_dict = mydict["treetagger_dict"]
    # load the pipeline
    # treetagger does only tokenization for some languages and pos, lemma
    annotated = mtt.MyTreetagger(treetagger_dict)
    # apply pipeline to data
    annotated.apply_to(data)
    # we should not need start ..?
    start = 0
    # for treetagger we always have list data as data will already be sentencized
    out_obj = mtt.OutTreetagger(annotated.doc, annotated.jobs, start=start, style=style)
    return out_obj


def call_flair(mydict, data, islist=True, style="STR"):
    flair_dict = mydict["flair_dict"]
    # load the pipeline
    # flair does only pos and ner
    annotated = mf.MyFlair(flair_dict)
    # apply pipeline to data
    # here we need to apply to each sentence one by one
    doc = []
    for sentence in data:
        annotated.apply_to(sentence)
        doc.append(annotated.doc)
    # we should not need start ..?
    start = 0
    print(annotated.jobs)
    # for flair we always have list data as data will already be sentencized
    out_obj = mf.OutFlair(doc, annotated.jobs, start=start, style=style)
    return out_obj


call_tool = {
    "spacy": call_spacy,
    "stanza": call_stanza,
    "somajo": call_somajo,
    "treetagger": call_treetagger,
    "flair": call_flair,
}


def run(path_json, path_txt):
    # load input dict
    mydict = be.PrepareRun.load_input_dict(path_json)
    # get the data to be processed
    data = be.PrepareRun.get_text(path_txt)
    # validate the input dict
    be.PrepareRun.validate_input_dict(mydict)
    # activate the input dict
    pe.SetConfig(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
    data_islist = False
    ptags = None
    stags = None
    if mydict["output_format"] == "vrt":
        style = "STR"
    elif mydict["output_format"] == "xml":
        style = "DICT"
    else:
        raise ValueError("Specified output format not recognized!")
    # we need ordered "set"
    tools = set()  # a temporary lookup set
    ordered_tools = [
        mytool
        for mytool in mydict["tool"]
        if mytool not in tools and tools.add(mytool) is None
    ]
    for mytool in ordered_tools:
        # here we do the object generation
        # we do not want to call same tools multiple times
        # as that would re-run the nlp pipelines
        # call specific routines
        my_out_obj = call_tool[mytool](mydict, data, data_islist, style)
        if not data_islist:
            # the first tool will sentencize
            # all subsequent ones will use sentencized input
            # so the new data is sentences from first tool
            # however, this is now a list
            data = my_out_obj.sentences
            # do the sentence-level processing
            # assemble sentences and tokens - this is independent of tool
            out = my_out_obj.assemble_output_sent()
            # further annotation: done with same tool?
            if mydict["tool"].count(mytool) > 2:
                print("Further annotation with tool {} ...".format(mytool))
                out = my_out_obj.assemble_output_tokens(out)
                ptags_temp = my_out_obj.ptags
                if ptags is not None:
                    ptags += ptags_temp
                else:
                    ptags = ptags_temp
            data_islist = True
            stags = my_out_obj.stags
        elif data_islist:
            # sentencized and tokenized data already processed
            # now token-level annotation
            # we need to keep a copy of token-list only for multi-step annotation
            # so that not of and of  ADP are being compared
            # or only compare to substring from beginning of string
            out = my_out_obj.assemble_output_tokens(out)
            ptags_temp = my_out_obj.ptags
            if ptags is not None:
                ptags += ptags_temp
            else:
                ptags = ptags_temp
    print(mydict["advanced_options"]["output_dir"])
    print(mydict["corpus_name"])
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    if style == "STR":
        # write out to .vrt
        be.OutObject.write_vrt(outfile, out)
    elif style == "DICT":
        # write out to .xml
        be.OutObject.write_xml(mydict["corpus_name"], outfile, out)
    # we will skip the encoding for now and instead provide vrt/xml file for user to download
    if mydict["encoding"] == "yes":
        print("Encoding the corpus into cwb...")
        encode_obj = be.EncodeCorpus(mydict)
        encode_obj.encode_vrt(ptags, stags)


if __name__ == "__main__":
    path_json = "./nlpannotator/data/input.json"
    path_txt = "./nlpannotator/test/data/example_en.txt"
    run(path_json, path_txt)
