# the base class and utilities are contained in this module
import json
from logging import raiseExceptions
import os
from numpy import string_


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

    # load the dictionary
    @staticmethod
    def load_input_dict(name):
        with open("{}.json".format(name)) as f:
            mydict = json.load(f)
        return mydict

    @staticmethod
    def update_dict(dict_in: dict) -> dict:
        """Remove unnecessary keys in dict and move processor-specific keys one level up."""
        # remove all comments - their keys start with "_"
        # also do not select sub-dictionaries
        dict_out = {k: v for k, v in dict_in.items() if not k.startswith("_")}
        return dict_out

    @staticmethod
    def activate_procs(mydict: dict, toolstring: str) -> dict:
        """Move processor-specific keys for a specific tool one level up.

        Args:
                mydict[dict]: Complete input dictionary.
                toolstring[str]: Indicates tool to activate entries for.

        Returns:
                [dict]: Dictionary containing input parameters for specific tool.
        """

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
    Assumes that there is one layer (no nested <> </> statements) of text elements to be separated.

    Args:
            path[str]: Path to .vrt file to be chunked.

    Returns:
            [list]: List containing the individual chunks.
    """

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
    next chunk after chunking the corpus. This method is currently not needed.

    Args:
            chunk[list]: List containing the lines for the .vrt as strings.
    """

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
        """Assemble the output after applying a pipeline to data.

        Args:
                doc: Doc-object returned by tool after applying pipeline to data.
                jobs[list]: List containing the specifiers for applied processors.
                start[int]: Starting Corpus-index for data.

        Returns:
                [list]: List containing the annotated .vrt lines.
        """

        obj = cls(doc, jobs, start)
        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence
        # if only sentence is provided, directly call the methods
        out = []
        if "sentence" not in obj.attrnames:
            raise KeyError("Error: Sentence-Key not in obj.attrnames.")

        obj.tstart = 0
        for sent in getattr(obj.doc, obj.attrnames["sentence"]):
            out.append("<s>\n")
            out = obj.iterate(out, sent)
            out.append("</s>\n")
        return out

    @staticmethod
    def get_names() -> dict:
        """Map the processors of different tools to the attribute names."""
        mydict = prepare_run.load_input_dict("attribute_names")
        return mydict

    def collect_results(self, token, tid, word, out: list) -> tuple:
        """Function to collect requested tags for tokens after applying pipeline to data."""

        # always get token id and token text
        # line = str(tid + start) + " " + token.text
        line = token.text
        # grab the data for the run components, I've only included the human readable
        # part of output right now as I don't know what else we need
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

    # we should have an option for vrt and one for xml writing
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
    def grab_ruler(token, tid: int, out, line: str):
        # attributes:
        # EntityRuler -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            line += "\t" + token.ent_type_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_linker(token, tid: int, out, line: str):
        # attributes:
        # EntityLinker -> Token.ent_kb_id, Token.ent_kb_id_
        if token.ent_type_ != "":
            line += "\t" + token.ent_kb_id_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_lemma(token, tid: int, word, out, line: str, attrname: str):
        # attributes:
        # spacy
        # Lemmatizer -> Token.lemma, Token.lemma_
        if word.lemma != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_morph(token, tid: int, out, line: str):
        # attributes:
        # Morphologizer -> Token.pos, Token.pos_, Token.morph
        if token.pos_ != "":
            line += "\t" + token.pos_ + "" + token.morph
        elif token.pos_ == "":
            line += "\t-" + token.morph
        return out, line

    @staticmethod
    def grab_tag(token, tid: int, word, out, line: str, attrname: str):
        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if getattr(word, attrname) != "":
            line += "\t" + getattr(word, attrname)
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_dep(token, tid: int, out, line: str):
        # attributes:
        # Parser -> Token.dep, Token.dep_, Token.head, Token.is_sent_start
        if token.dep_ != "":
            line += "\t" + token.dep_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def grab_att(token, tid: int, out, line: str):
        # attributes:
        if token.pos_ != "":
            line += "\t" + token.pos_
        else:
            line += "\t-"
        return out, line

    @staticmethod
    def write_vrt(outname: str, out: list):
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

    def __init__(self, corpusname: str, outname: str, jobs: list, tool: str) -> None:
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

    def _get_s_attributes(self, line: str) -> str:
        if any(attr in self.attrnames["proc_sent"] for attr in self.jobs):
            print("Encoding s-attribute <s>...")
            line += "-S s "
        return line

    # the order here is important for vrt files and MUST NOT be changed!!!
    def _get_p_attributes(self, line: str) -> str:
        if any(attr in self.attrnames["proc_pos"] for attr in self.jobs):
            print("Encoding p-attribute POS...")
            line += "-P pos "
        if any(attr in self.attrnames["proc_lemma"] for attr in self.jobs):
            print("Encoding p-attribute lemma...")
            line += "-P lemma "
        return line

    @classmethod
    def encode_vrt(cls, corpusname: str, outname: str, jobs: list, tool: str):
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
