# the base class and utilities are contained in this module
import json
from logging import raiseExceptions
import os
import copy


# a dictionary to map attribute names for the different tools that we use
dictmap = {
    "stanza_names": {
        "proc_lemma": "lemma",
        "proc_pos": "pos",
        "sentence": "sentences",
        "token": "tokens",
        "pos": "upos",
        "lemma": "lemma",
        "ner": "text",
    },
    "spacy_names": {
        "proc_lemma": "lemmatizer",
        "proc_pos": "attribute_ruler",
        "sentence": "sents",
        "token": "token",
        "pos": "pos_",
        "lemma": "lemma_",
        "ner": "ent_type_",
    },
}


# the below functions in a class with attributes
def get_cores() -> int:
    """Find out how many CPU-cores are available for current process."""
    return len(os.sched_getaffinity(0))


# the below functions will move into an input class
# read the sample data
def get_sample_text():
    name = "data/Original/iued_test_original.txt"
    with open(name, "r") as myfile:
        data = myfile.read().replace("\n", "")
    return data


# I thought this would belong here rather than mspacy.
def chunk_sample_text(path: str) -> list:
    """Function to chunk down a given vrt file into pieces sepparated by <> </> boundaries.
    Assumes that there is one layer (no nested <> </> statements) of text elements to be separated."""
    # list for data chunks
    data = []
    # index to refer to current chunk
    i = 0
    # bool to set if we are currently in paragraph or inbetween
    inpar = False

    with open(path, "r") as myfile:
        # iterate .vrt
        for line in myfile:
            # if line starts with "<":
            if line.startswith("<"):
                # if we are not in paragraph and not in chunk already:
                if inpar is False:
                    # we are now in paragraph
                    inpar = True
                    # add chunk to list-> chunk is list of three strings:
                    # chunk[0]: Opening "<>" statement
                    # chunk[1]: Text contained in chunk, every "\n" replaced with " "
                    # chunk[2]: Next "<>" statement
                    data.append(["", "", ""])
                    data[i][0] += line.replace("\n", " ")

                # if we are in paragraph and not in subchunk
                elif inpar is True:
                    # we are no longer in paragraph
                    inpar = False
                    # add end statement to chunk -> start new chunk next iteration
                    data[i][2] += line.replace("\n", " ")
                    # increment chunk idx
                    i += 1

            # if we are in paragraph:
            elif inpar:
                # append line to chunk[1], replacing "\n" with " "
                data[i][1] += line.replace("\n", " ")

    return data


def find_last_idx(chunk: list) -> int:
    """Function to find last index in chunk to keep token index up to date for
    next chunk after chunking the corpus.

    [Args]:
            chunk[list]: List containing the lines for the .vrt as strings."""
    # get the index to last element
    i = len(chunk) - 1
    # iterate through entire chunk if neccessary, should never happen in practice
    for j in range(len(chunk)):
        # if string starts with "<" last elem isnt line string but some s-attribute
        if chunk[i].split()[0].startswith("<"):
            # set index to next element
            i -= 1
        else:
            # if string doesnt start with "<" we can assume it contains the token index
            # in the first column
            return int(chunk[i].split()[0])


# load the dictionary
def load_input_dict(name):
    with open("src/annotator/{}.json".format(name)) as f:
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


# now this becomes the base out_object class
# this class is inherited in the different modules
# selected methods are overwritten/added depending on the requirements
# the mapping dict will remain to make the conversion clear and not to duplicate
class out_object:
    """The base output object and namespace. Write the vrt file."""

    def __init__(self) -> None:
        # get the attribute names for the different tools
        self.attrnames = load_input_dict("attribute_names")

    @staticmethod
    def open_outfile(outname):
        name = "out/" + outname
        f = open(name, "w")
        return f

    @staticmethod
    def assemble_output_sent(doc, inpname, jobs, start, tool):
        # get the dictionary map for the attribute names that are unique to each tool
        attrnames = out_object._get_names(tool)

        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence
        out = []

        # spacy
        # for sent in doc.sents:
        # stanza
        # for sent in doc.sentences:
        # general
        # count stanza tokens continuously and not starting from 1 every new sentence.
        tstart = 0
        for sent in getattr(doc, attrnames["sentence"]):
            out.append("<s>\n")
            # iterate through the tokens of the sentence, this is just a slice of
            # the full doc
            # spacy
            # for token in sent:
            # stanza
            # for multi-word tokens this could prove problematic
            # we have to distinguish btw token and word in that case
            # for token, word in zip(sent.tokens, sent.words):
            # general
            if tool == "spacy":
                out = out_object._iterate_spacy(out, sent, attrnames, jobs, start)
            elif tool == "stanza":
                out, tstart = out_object._iterate_stanza(
                    out, sent, attrnames, jobs, start, tstart
                )
            else:
                raise NotImplementedError(
                    "Tool {} not available at this time.".format(tool)
                )
        out[1] += " \n"
        return out

    def _get_names(tool) -> dict:
        mydict = dictmap[tool + "_names"]
        return mydict

    def _iterate_spacy(out, sent, attrnames, jobs, start):
        for token in sent:
            # multi-word expressions not available in spacy?
            # Setting word=token for now
            tid = copy.copy(token.i)
            out, line = out_object.collect_results(
                jobs, token, tid, token, out, attrnames, start=start
            )
            out.append(line + "\n")
        out.append("</s>\n")
        return out

    def _iterate_stanza(out, sent, attrnames, jobs, start, tstart):
        for token, word in zip(getattr(sent, "tokens"), getattr(sent, "words")):
            if token.text != word.text:
                raise NotImplementedError(
                    "Multi-word expressions not available currently"
                )
            tid = token.id[0] + tstart
            # for ent in getattr(sent, "ents"):
            # print(ent)
            out, line = out_object.collect_results(
                jobs, token, tid, word, out, attrnames, start=start
            )
            out.append(line + "\n")
        out.append("</s>\n")
        tstart = tid
        return out, tstart

    def collect_results(jobs, token, tid, word, out: list, attrnames, start=0) -> tuple:
        """Function to collect requested tags for tokens after applying pipeline to data."""
        # always get token id and token text
        # line = str(tid + start) + " " + token.text
        line = token.text
        # grab the data for the run components, I've only included the human readable
        # part of output right now as I don't know what else we need
        ########
        # we need to unify the names for the different job types
        # ie spacy - lemmatizer, stanza - lemma
        # spacy - tagger, stanza - pos
        # spacy - ner, stanza - ner
        # have to find out how ner is encoded in cwb first
        #########
        # put in correct order - first pos, then lemma
        # order matters for encoding

        if attrnames["proc_pos"] in jobs:
            out, line = out_object.grab_tag(
                token, tid, word, out, line, attrnames["pos"]
            )

        if attrnames["proc_lemma"] in jobs:
            out, line = out_object.grab_lemma(
                token, tid, word, out, line, attrnames["lemma"]
            )

        if "ner" in jobs:
            out, line = out_object.grab_ner(token, tid, out, line)

        if "entity_ruler" in jobs:
            out, line = out_object.grab_ruler(token, tid, out, line)

        if "entity_linker" in jobs:
            out, line = out_object.grab_linker(token, tid, out, line)

        if "morphologizer" in jobs:
            out, line = out_object.grab_morph(token, tid, out, line)

        if "parser" in jobs:
            out, line = out_object.grab_dep(token, tid, out, line)

        if "attribute_ruler" in jobs:
            out, line = out_object.grab_att(token, tid, out, line)
        # add what else we need

        return out, line

    # define all of these as functions
    def grab_ner(token, tid, out, line):
        # attributes:
        # EntityRecognizer -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            line += "\t" + token.ent_type_
        else:
            line += "\t-"
        return out, line

    def grab_ruler(token, tid, out, line):
        # attributes:
        # EntityRuler -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            line += "\t" + token.ent_type_
        else:
            line += "\t-"
        return out, line

    def grab_linker(token, tid, out, line):
        # attributes:
        # EntityLinker -> Token.ent_kb_id, Token.ent_kb_id_
        if token.ent_type_ != "":
            line += "\t" + token.ent_kb_id_
        else:
            line += "\t-"
        return out, line

    def grab_lemma(token, tid, word, out, line, attrname):
        # attributes:
        # spacy
        # Lemmatizer -> Token.lemma, Token.lemma_
        if word.lemma != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    def grab_morph(token, tid, out, line):
        # attributes:
        # Morphologizer -> Token.pos, Token.pos_, Token.morph
        if token.pos_ != "":
            line += "\t" + token.pos_ + "" + token.morph
        elif token.pos_ == "":
            line += "\t-" + token.morph
        return out, line

    def grab_tag(token, tid, word, out, line, attrname):
        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if getattr(word, attrname) != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    def grab_dep(token, tid, out, line):
        # attributes:
        # Parser -> Token.dep, Token.dep_, Token.head, Token.is_sent_start
        if token.dep_ != "":
            line += "\t" + token.dep_
        else:
            line += "\t-"
        return out, line

    def grab_att(token, tid, out, line):
        # attributes:
        if token.pos_ != "":
            line += "\t" + token.pos_
        else:
            line += "\t-"
        return out, line

    def to_vrt(outname, out) -> list or None:
        """Function to write list to a .vrt file.

        [Args]:
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
        """
        with open("{}.vrt".format(outname), "w") as file:
            for line in out:
                file.write(line)
        print("+++ Finished writing .vrt +++")


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
