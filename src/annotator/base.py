# the base class and utilities are contained in this module
import json


# read the sample data
def get_sample_text():
    name = "data/Original/iued_test_original.txt"
    with open(name, "r") as myfile:
        data = myfile.read().replace("\n", "")
    return data


# load the dictionary
def load_input_dict():
    with open("src/annotator/input.json") as f:
        mydict = json.load(f)
    return mydict


def update_dict(dict_in) -> dict:
    """Remove unnecessary keys in dict and move processor-specific keys one level up."""
    # remove all comments - their keys start with "_"
    # also do not select sub-dictionaries
    dict_out = {k: v for k, v in dict_in.items() if not k.startswith("_")}
    return dict_out


# open outfile
def open_outfile():
    name = "out/output.txt"
    f = open(name, "w")
    return f


class out_object:
    """The output object. Write the vrt file."""

    def __init__(self) -> None:
        pass

    # define all of these as functions
    def grab_ner(token, out, line):
        if token.i == 0:
            out[1] += " ner"
        if token.ent_type_ != "":
            line += "  " + token.ent_type_
        else:
            line += " - "
        return out, line

    def grab_lemma(token, out, line):
        if token.i == 0:
            out[1] += " lemma"
        if token.lemma_ != "":
            line += " " + token.lemma_
        else:
            line += " - "
        return out, line

    def grab_tag(token, out, line):
        if token.i == 0:
            out[1] += " Tag"
        if token.tag_ != "":
            line += " " + token.tag_
        else:
            line += " - "
        return out, line

    def grab_dep(token, out, line):
        if token.i == 0:
            out[1] += " Depend"
        if token.dep_ != "":
            line += " " + token.dep_
        else:
            line += " - "
        return out, line

    def grab_att(token, out, line):
        if token.i == 0:
            out[1] += " POS"
        if token.pos_ != "":
            line += " " + token.pos_
        else:
            line += " - "
        return out, line


# metadata and tags
# metadata at top of document
# <corpus>
# <document>
#   <metadata>
#       <author></author>
#       <speaker_name>
#       <speaker_party>
#       <speaker_role>
#       <lp>
#       <session>
#       <date>
#       <year>
#       <year_month>
#       <speaker_next>
#        ...
#       <text>
#   </metadata>
# now the annotated text
# <text id="" speaker_name="" ...>
#  - main text with s-attributes and attributes
# <sp> speech
# <z> Zwischenrufe
# <s id=""> sentence
# <t id=""> token
# <pt> <numb><t id=""> thirteen ..
# <pt><prop><t id=""> Audi ..
# <pt><comp> ..
# <pt><emb><noun>
#
#
#
#
