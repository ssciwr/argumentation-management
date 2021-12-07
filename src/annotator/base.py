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
        dict = json.load(f)
    return dict


# open outfile
def open_outfile():
    name = "out/output.txt"
    f = open(name, "w")
    return f


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
