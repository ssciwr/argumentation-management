# the base class and utilities are contained in this module
import json
from logging import raiseExceptions
import os

from numpy import string_


# the below functions in a class with attributes
class prepare_run:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_cores() -> int:
        """Find out how many CPU-cores are available for current process."""
        # will need to update this to using multiprocess
        # as this method is not available on all os's
        return len(os.sched_getaffinity(0))

    # read the sample data - this will be in test_base
    # method will be removed
    @staticmethod
    def get_sample_text():
        name = "data/Original/iued_test_original.txt"
        with open(name, "r") as myfile:
            data = myfile.read().replace("\n", "")
        return data

    @staticmethod
    def get_text(path: str) -> str:
        with open(path, "r") as input:
            data = input.read().replace("\n", "")
        return data

    # load the dictionary
    @staticmethod
    def load_input_dict(name):
        with open("{}.json".format(name)) as f:
            mydict = json.load(f)
        return mydict

    @staticmethod
    def update_dict(dict_in) -> dict:
        """Remove unnecessary keys in dict and move processor-specific keys one level up."""
        # remove all comments - their keys start with "_"
        # also do not select sub-dictionaries
        dict_out = {k: v for k, v in dict_in.items() if not k.startswith("_")}
        return dict_out

    @staticmethod
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


# the below in a chunker class
# I thought this would belong here rather than mspacy.
def chunk_sample_text(path: str) -> list:
    """Function to chunk down a given vrt file into pieces sepparated by <> </> boundaries.
    Assumes that there is one layer (no nested <> </> statements) of text elements to be separated."""
    # list for data chunks
    data = []
    # index to refer to current chunk
    i = 0
    # index of seen xml elements
    xml_seen = 0
    with open(path, "r") as myfile:

        # iterate .vrt
        for line in myfile:
            # if line starts with "<" and sml seen == 0 we have the first chunk
            if line.startswith("<") and xml_seen == 0:
                # we have now seen an xml element
                xml_seen += 1
                # add chunk to list-> chunk is list of three strings:
                # chunk[0]: Opening "<>" statement
                # chunk[1]: Text contained in chunk, every "\n" replaced with " "
                # chunk[2]: Next "<>" statement
                data.append(["", "", ""])
                data[i][0] += line.replace("\n", " ")

            elif line.startswith("<") and xml_seen > 0:
                # we've seen another one
                xml_seen += 1
                # if we encounter a closing statement we end the current chunk
                if line.startswith("</"):

                    data[i][2] = line.replace("\n", " ")
                    i += 1
                    data.append(["", "", ""])

                # else we encountered another opening xml element and are in a nested environment
                # we also start a new chunk but leave the closing statement of the previous one empty
                else:

                    i += 1
                    data.append(["", "", ""])
                    data[i][0] = line.replace("\n", " ")

            # if we are not on a line with an xml element we can just write the text to the
            # text entry (idx 1) for the current chunk, "inter-chunk indexing" should be handled
            # by the above case selection
            else:
                # append line to chunk[1], replacing "\n" with " "
                data[i][1] += line.replace("\n", " ")

    # if we appended empty chunks we remove them here
    for chunk in data:
        if all(elems == "" for elems in chunk):
            data.remove(chunk)

    # we should be able to check for validity here -> can maybe outsource this to a function later
    if xml_seen % 2 != 0:
        # the number of xml elements in a valid document should always be even
        raise RuntimeError("Encountered uneven number of XML elements in imput!")
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
            print(chunk[i].split()[0])
            return int(chunk[i].split()[0])


# the base out_object class
# this class is inherited in the different modules
# selected methods are overwritten/added depending on the requirements
# the mapping dict to make the conversion clear and not to duplicate code
class out_object:
    """The base output object and namespace. Write the vrt file."""

    def __init__(self, doc, jobs, start) -> None:
        self.doc = doc
        self.jobs = jobs
        self.start = start
        # get the attribute names for the different tools
        self.attrnames = self.get_names()

    @staticmethod
    def open_outfile(outname):
        name = "out/" + outname
        f = open(name, "w")
        return f

    @classmethod
    def assemble_output_sent(cls, doc, jobs, start):
        obj = cls(doc, jobs, start)
        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence
        # if only sentence is provided, directly call the methods
        out = []
        # spacy
        # for sent in doc.sents:
        # stanza
        # for sent in doc.sentences:
        # general
        # count stanza tokens continuously and not starting from 1 every new sentence.
        if "sentence" not in obj.attrnames:
            raise KeyError("Error: Sentence-Key not in obj.attrnames.")

        obj.tstart = 0
        for sent in getattr(obj.doc, obj.attrnames["sentence"]):
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
            out = obj.iterate(out, sent)
            out.append("</s>\n")
        return out

    @staticmethod
    def get_names() -> dict:
        mydict = prepare_run.load_input_dict("attribute_names")
        # mydict = prepare_run.load_input_dict("src/annotator/attribute_names")
        return mydict

    def collect_results(self, token, tid, word, out: list) -> tuple:
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

        if self.attrnames["proc_pos"] in self.jobs:
            out, line = out_object.grab_tag(
                token, tid, word, out, line, self.attrnames["pos"]
            )

        if self.attrnames["proc_lemma"] in self.jobs:
            out, line = out_object.grab_lemma(
                token, tid, word, out, line, self.attrnames["lemma"]
            )

        if "ner" in self.jobs:
            out, line = out_object.grab_ner(token, tid, out, line)

        if "entity_ruler" in self.jobs:
            out, line = out_object.grab_ruler(token, tid, out, line)

        if "entity_linker" in self.jobs:
            out, line = out_object.grab_linker(token, tid, out, line)

        if "morphologizer" in self.jobs:
            out, line = out_object.grab_morph(token, tid, out, line)

        if "parser" in self.jobs:
            out, line = out_object.grab_dep(token, tid, out, line)

        if "attribute_ruler" in self.jobs:
            out, line = out_object.grab_att(token, tid, out, line)
        # add what else we need

        return out, line

    # define all of these as functions
    # these to be either internal or static methods
    # we should have an option for vrt and one for xml writing
    # making them static for now
    @staticmethod
    def grab_ner(token, tid, out, line):
        # attributes:
        # EntityRecognizer -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            line += "\t" + token.ent_type_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_ruler(token, tid, out, line):
        # attributes:
        # EntityRuler -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            line += "\t" + token.ent_type_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_linker(token, tid, out, line):
        # attributes:
        # EntityLinker -> Token.ent_kb_id, Token.ent_kb_id_
        if token.ent_type_ != "":
            line += "\t" + token.ent_kb_id_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_lemma(token, tid, word, out, line, attrname):
        # attributes:
        # spacy
        # Lemmatizer -> Token.lemma, Token.lemma_
        if word.lemma != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_morph(token, tid, out, line):
        # attributes:
        # Morphologizer -> Token.pos, Token.pos_, Token.morph
        if token.pos_ != "":
            line += "\t" + token.pos_ + "" + token.morph
        elif token.pos_ == "":
            line += "\t-" + token.morph
        return out, line

    @staticmethod
    def grab_tag(token, tid, word, out, line, attrname):
        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if getattr(word, attrname) != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_dep(token, tid, out, line):
        # attributes:
        # Parser -> Token.dep, Token.dep_, Token.head, Token.is_sent_start
        if token.dep_ != "":
            line += "\t" + token.dep_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_att(token, tid, out, line):
        # attributes:
        if token.pos_ != "":
            line += "\t" + token.pos_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def write_vrt(outname, out):
        """Function to write list to a .vrt file.

        [Args]:
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
        """
        with open("{}.vrt".format(outname), "w") as file:
            for line in out:
                file.write(line)
        print("+++ Finished writing {}.vrt +++".format(outname))


# encode the generated files
class encode_corpus:
    """Encode the vrt/xml files for cwb."""

    def __init__(self, corpusname, outname, jobs, tool) -> None:
        # self.corpusdir = "/home/jovyan/corpus"
        self.corpusdir = "/home/inga/projects/corpus-workbench/cwb/corpora/"
        self.corpusname = corpusname
        self.outname = outname
        # self.regdir = "/home/jovyan/registry"
        self.regdir = "/home/inga/projects/corpus-workbench/cwb/registry/"
        self.jobs = jobs
        self.tool = tool
        self.encodedir = self.corpusdir + corpusname
        # create the new corpus' directory if not there yet
        try:
            os.makedirs(self.encodedir)
        except OSError:
            pass
        # get attribute names
        self.attrnames = out_object.get_names()
        self.attrnames = self.attrnames[self.tool + "_names"]

    def _get_s_attributes(self, line) -> str:
        if any(attr in self.attrnames["proc_sent"] for attr in self.jobs):
            print("Encoding s-attribute <s>...")
            line += "-S s "
        return line

    # the order here is important for vrt files and MUST NOT be changed!!!
    def _get_p_attributes(self, line) -> str:
        if any(attr in self.attrnames["proc_pos"] for attr in self.jobs):
            print("Encoding p-attribute POS...")
            line += "-P pos "
        if any(attr in self.attrnames["proc_lemma"] for attr in self.jobs):
            print("Encoding p-attribute lemma...")
            line += "-P lemma "
        return line

    @classmethod
    def encode_vrt(cls, corpusname, outname, jobs, tool):
        obj = cls(corpusname, outname, jobs, tool)
        line = " "
        # find out which options are to be encoded
        line = obj._get_s_attributes(line)
        line = obj._get_p_attributes(line)
        # call the os with the encode command
        print("Encoding the corpus...")
        print("Options are:")
        command = (
            "cwb-encode -d "
            + obj.encodedir
            + " -xsBC9 -c ascii -f "
            + obj.outname
            + ".vrt -R "
            + obj.regdir
            + obj.corpusname
            + line
        )
        print(command)
        os.system(command)
        print("Updating the registry entry...")
        print("Options are:")
        # call the os with the registry update command
        command = "cwb-makeall -r " + obj.regdir + " -V " + obj.corpusname
        print(command)
        os.system(command)


# en
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
