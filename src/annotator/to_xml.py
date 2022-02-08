from xml.etree.ElementTree import Element, tostring
import xml.dom.minidom as mini
import mstanza as ms
import base as be


def dict_to_xml(tag, d):

    elem = Element(tag)
    for key, val in d.items():
        child = Element(key)
        child.text = str(val)
        elem.append(child)

    return elem


if __name__ == "__main__":
    dict = be.prepare_run.load_input_dict("./src/annotator/input")
    outfile = dict["output"]
    # take only the part of dict pertaining to stanza
    stanza_dict = dict["stanza_dict"]
    # to point to user-defined model directories
    # stanza does not accommodate fully at the moment
    mydict = ms.mstanza_preprocess.fix_dict_path(stanza_dict)
    # stanza does not care about the extra comment keys
    # but we remove them for subsequent processing just in case
    # now we need to select the processors and "activate" the sub-dictionaries
    mydict = be.prepare_run.update_dict(mydict)
    mydict = be.prepare_run.activate_procs(mydict, "stanza_")
    mytext = be.prepare_run.get_sample_text()
    # mytext = "This is an example. And here we go."
    # initialize instance of the class
    obj = ms.mstanza_pipeline(mydict)
    obj.init_pipeline()
    out = obj.process_text(mytext)
    # obj.postprocess(outfile)

    dicts = out.to_dict()

    out_xml = ""

    for i, elem in enumerate(dicts, 1):
        for j, el in enumerate(elem, 1):
            out_xml += tostring(
                dict_to_xml("SentID={}, TokID={}".format(i, j), el)
            ).decode()

    with open("test.xml", "w") as file:
        file.write(out_xml)

    parsed = mini.parse("test.xml")

    print(parsed.toprettyxml())
    # For the output:
    # We need a module that transforms a generic dict into xml.
