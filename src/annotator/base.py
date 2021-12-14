# the base class and utilities are contained in this module
import json


# the below functions will move into an input class
# read the sample data
def get_sample_text():
    name = "data/Original/iued_test_original.txt"
    with open(name, "r") as myfile:
        data = myfile.read().replace("\n", "")
    return data


# I thought this would be belong here rather than mspacy.
# try to chunk the plenary text from example into pieces, annotate these and than reasemble to .vrt
def chunk_sample_text(path):
    # list for data chunks
    data = []
    # index to refer to current chunk
    i = 0
    # bool to set if we are currently in paragraph or inbetween
    inpar = False
    # counter to identify sublayers (<>text<>text</></>). This could be used to iteratively chunk down a corpus by
    # calling this function on the chunks until no more sublayers are present
    sub = 0
    layers = False

    with open(path, "r") as myfile:
        # iterate .vrt
        for line in myfile:
            # if line starts with "<":
            if line[0] == "<" and line[1] != "/":
                # if we are not in paragraph and not in chunk already:
                if inpar is False and sub == 0:
                    # we are now in paragraph
                    inpar = True
                    # add chunk to list-> chunk is list of three strings:
                    # chunk[0]: Opening "<>" statement
                    # chunk[1]: Text contained in chunk, every "\n" replaced with " "
                    # chunk[2]: Next "<>" statement
                    data.append(["", "", ""])
                    data[i][0] += line.replace("\n", " ")
                elif sub > 0:
                    data[i][1] += line.replace("\n", " ")
                # we are one layer deeper in every opening <> statement will be chunk
                sub += 1

            elif line[0] == "<" and line[1] == "/":

                sub -= 1
                # if we are in paragraph and not in subchunk
                if inpar is True and sub == 0:
                    # we are no longer in paragraph
                    inpar = False
                    # add end statement to chunk -> start new chunk next iteration
                    data[i][2] += line.replace("\n", " ")
                    # increment chunk idx
                    i += 1
                elif inpar is True and sub > 0:
                    data[i][1] += line.replace("\n", " ")
            # if we are in paragraph:
            elif inpar:
                # append line to chunk[1], replacing "\n" with " "
                data[i][1] += line.replace("\n", " ")

            if sub > 1:
                layers = True

    return data, layers


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


def activate_procs(mydict, toolstring) -> dict:
    """Move processor-specific keys one level up."""
    # find out which processors were selected
    procs = mydict.get("processors", None)
    if procs is None:
        raise ValueError("Error: No processors defined!")
    # separate the processor list at the comma
    procs = procs.split(",")
    # pick the corresponding dictionary and clean comments
    for proc in procs:
        mystring = toolstring + proc
        mydict.update(
            {k: v for k, v in mydict[mystring].items() if not k.startswith("_")}
        )
    # remove all other processor dictionaries that are not used
    # this is not really necessary for stanza but doing it to keep the dict clean
    mydict = {k: v for k, v in mydict.items() if not k.startswith("toolstring")}
    return mydict


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
        # attributes:
        # EntityRecognizer -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.i == 0:
            out[1] += " ner"
        if token.ent_type_ != "":
            line += "  " + token.ent_type_
        else:
            line += " - "
        return out, line

    def grab_ruler(token, out, line):
        # attributes:
        # EntityRuler -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.i == 0:
            out[1] += " entity_ruler"
        if token.ent_type_ != "":
            line += "  " + token.ent_type_
        else:
            line += " - "
        return out, line

    def grab_linker(token, out, line):
        # attributes:
        # EntityLinker -> Token.ent_kb_id, Token.ent_kb_id_
        if token.i == 0:
            out[1] += " entity_linker"
        if token.ent_type_ != "":
            line += "  " + token.ent_kb_id_
        else:
            line += " - "
        return out, line

    def grab_lemma(token, out, line):
        # attributes:
        # Lemmatizer -> Token.lemma, Token.lemma_
        if token.i == 0:
            out[1] += " lemma"
        if token.lemma_ != "":
            line += " " + token.lemma_
        else:
            line += " - "
        return out, line

    def grab_morph(token, out, line):
        # attributes:
        # Morphologizer -> Token.pos, Token.pos_, Token.morph
        if token.i == 0:
            out[1] += " UPOS morph"
        if token.lemma_ != "":
            line += " " + token.pos_ + "" + token.morph
        else:
            line += " - "
        return out, line

    def grab_tag(token, out, line):
        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if token.i == 0:
            out[1] += " Tag"
        if token.tag_ != "":
            line += " " + token.tag_
        else:
            line += " - "
        return out, line

    def grab_dep(token, out, line):
        # attributes:
        # Parser -> Token.dep, Token.dep_, Token.head, Token.is_sent_start
        if token.i == 0:
            out[1] += " pars"
        if token.dep_ != "":
            line += " " + token.dep_
        else:
            line += " - "
        return out, line

    def grab_att(token, out, line):
        # attributes:
        # Token.pos, Token.pos_
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
