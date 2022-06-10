# the base class and utilities are contained in this module
import json
import jsonschema
import os
import to_xml as txml


class prepare_run:
    """Class that contains all general pre-processing methods."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_cores() -> int:
        """Find out how many CPU-cores are available for current process."""

        # will need to update this to using multiprocess
        # as this method is not available on all os's
        return len(os.sched_getaffinity(0))

    @staticmethod
    def get_text(path: str) -> str:
        """Convenience function to read in data from specified path as string.

        Args:
                path[str]: Path to data."""

        with open(path, "r") as input:
            data = input.read().replace("\n", "")
        return data

    # load the dictionary
    @staticmethod
    def load_input_dict(name: str) -> dict:
        """Function to load input dictionary from .json.

        Args:
                name[str]: Name of .json file (without file extension)."""

        with open("{}.json".format(name)) as f:
            mydict = json.load(f)
        return mydict

    # load the dictionary schema and validate against
    @staticmethod
    def validate_input_dict(dict_in: dict) -> None:
        with open(
            "{}.json".format("input_schema"),
            # "{}.json".format("./src/annotator/input_schema"),
            "r",
        ) as f:
            myschema = json.load(f)
        jsonschema.validate(instance=dict_in, schema=myschema)

    # @staticmethod
    # def get_encoding(dict_in: dict) -> dict:
    #     """Function to fetch the parameters needed for encoding from the input.json."""

    #     new_dict = {}

    #     for key, value in dict_in.items():

    #         if type(value) != dict or type(value) == dict and key == "advanced_options":
    #             new_dict[key] = value
    #         elif type(value) == dict and key == "advanced_options":
    #             new_dict[key] = value

    #     new_dict["processors"] = dict_in["{}_dict".format(dict_in["tool"])][
    #     "processors"
    #     ]

    #     return new_dict


# the below to be removed - TODO
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
            # print(chunk[i].split()[0])
            return int(chunk[i].split()[0])


# set the string to be used for undefined tags
NOT_DEF = " "


# the base OutObject class
# this class is inherited in the different modules
# selected methods are overwritten/added depending on the requirements
# the mapping dict to make the conversion clear and not to duplicate code
class OutObject:
    """The base output object and namespace. Write the vrt file."""

    def __init__(self, doc, jobs: list, start: int, islist=False):
        self.doc = doc
        self.jobs = jobs
        self.start = start
        # just one doc object for whole text or multiple objects per sentence
        # (self.doc)
        self.islist = islist
        # get the attribute names for the different tools
        self.attrnames = self.get_names()

    @staticmethod
    def open_outfile(outname: str):
        """Initialize output file.

        Args:
                outname[str]: Name of file to be created in out directory."""

        name = "out/" + outname
        f = open(name, "w")
        return f

    def iterate(self, out, sent, style):
        # this needs to be done module-specific for now and is set in each subclass
        pass

    def assemble_output_sent(self) -> list:
        """Template function to assemble output for tool at sentence level."""

        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence
        # if only sentence is provided, directly call the methods
        if "sentence" not in self.attrnames:
            raise KeyError("Error: Sentence-Key not in obj.attrnames.")

        self.tstart = 0
        out = []
        for sent in getattr(self.doc, self.attrnames["sentence"]):
            out.append("<s>\n")
            # for token in sent:
            # self.out.append(token.text + "\n")
            out = self.iterate(out, sent, "STR")
            # self.out.append(sent.text + "\n")
            out.append("</s>\n")
        return out

    def assemble_output_tokens(self, out) -> list:
        """Template function to assemble output for tool at token level."""
        # this needs to be done module-specific for now and is set in each subclass
        return out

    def assemble_output_xml(self):
        out = []
        if "sentence" not in self.attrnames:
            raise KeyError("Error: Sentence-Key not in obj.attrnames.")
        self.tstart = 0
        for sent in getattr(self.doc, self.attrnames["sentence"]):
            out.append([])
            self.iterate(out[-1], sent, "DICT")
        return out

    @staticmethod
    def get_names() -> dict:
        """Load attribute names for specific tools."""

        mydict = prepare_run.load_input_dict("attribute_names")
        # mydict = prepare_run.load_input_dict("src/annotator/attribute_names")
        return mydict

    # refactor once STR is working - we actually do not need token text as key
    # token texts are handled when assembling the sentences
    @staticmethod
    def switch_style(line: dict) -> str:
        """Switch style from DICT to STR"""

        output = ""
        for i, (key, value) in enumerate(line.items()):
            # don't repeat the token, we already got it from sentencize/tokenize
            # a temporary fix, later we will not add this to the dict in the first place
            if key != "text":
                # strip the id, and don't append \t for the "first" item
                if key != "id" and i > 1:
                    output += "\t{}".format(value)
                elif key != "id" and i == 1:
                    output += "{}".format(value)
        return output

    # remove repetition - TODO
    def get_ptags(self) -> list or None:
        """Function to easily collect the current ptags in a list.

        !!!
        Does the same case selection as OutObject.collect_results, so the order
        of .vrt and this list should always be identical. If you change one
        MAKE SURE to also change the other.
        !!!
        """

        ptags = []

        if self.attrnames["proc_pos"] in self.jobs:
            ptags.append("pos")
        if self.attrnames["proc_lemma"] in self.jobs:
            ptags.append("lemma")
        if "ner" in self.jobs:
            ptags.append("NER")
        if "attribute_ruler" in self.jobs:
            ptags.append("ATTR")

        if ptags != []:
            return ptags
        else:
            return None

    def get_stags(self) -> list:

        stags = []
        if any(attr in self.attrnames["proc_sent"] for attr in self.jobs):
            stags.append("s")

        return stags

    def collect_results(self, token, tid: int, word, style: str = "STR") -> dict or str:

        """Function to collect requested tags for tokens after applying pipeline to data.

        Args:
                style[str]. Return line as string (STR) for .vrt or dict (DICT) for .xml."""

        # always get token id and token text
        # line = str(tid + start) + " " + token.text
        line = {"id": str(tid), "text": token.text}
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

            line["POS"] = OutObject.grab_tag(word, self.attrnames["pos"])

        if self.attrnames["proc_lemma"] in self.jobs:
            line["LEMMA"] = OutObject.grab_lemma(word, self.attrnames["lemma"])

        if "ner" in self.jobs:
            line["NER"] = OutObject.grab_ent(token)

        if style == "STR":

            return self.switch_style(line)

        elif style == "DICT":

            return line

    @staticmethod
    def grab_ent(token):

        # attributes:
        # EntityRecognizer -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type
        # EntityRuler -> Token_iob, Token.ent_iob_, Token.ent_type, Token.ent_type_
        if token.ent_type_ != "":
            tag = token.ent_type_
        else:
            tag = NOT_DEF
        return tag

    @staticmethod
    def grab_lemma(word, attrname):

        # attributes:
        # spacy
        # Lemmatizer -> Token.lemma, Token.lemma_
        if word.lemma != "":
            tag = getattr(word, attrname)
        else:
            tag = NOT_DEF
        return tag

    @staticmethod
    def grab_tag(word, attrname):

        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if getattr(word, attrname) != "":
            tag = getattr(word, attrname)
        else:
            tag = NOT_DEF
        return tag

    @staticmethod
    def purge(out_string: str) -> str:
        """Function to search and replace problematic patterns in tokens
        before encoding."""

        # expand these with more if neccessary, correct mapping is important!

        out_string = out_string.replace(" ", "")

        return out_string

    @staticmethod
    def write_vrt(outname: str, out: list) -> None:
        """Function to write list to a .vrt file.

        [Args]:
            out[list]: List containing the lines for the .vrt file as strings.
        """

        string = ""
        for line in out:
            string += line

        string = OutObject.purge(string)

        with open("{}.vrt".format(outname), "w") as file:
            file.write(string)

        print("+++ Finished writing {}.vrt +++".format(outname))

    @staticmethod
    def write_xml(docid: str, outname: str, out: list) -> None:
        """This function may not work for all tools and should not be used at the moment."""

        raw_xml = txml.start_xml(docid)

        sents = [txml.list_to_xml("Sent", i, elem) for i, elem in enumerate(out, 1)]

        for sent in sents:
            raw_xml.append(sent)

        string_xml = txml.to_string(raw_xml)

        xml = txml.beautify(string_xml)

        with open("{}_.xml".format(outname), "w") as file:
            file.write(xml)

        print("+++ Finished writing {}.xml +++".format(outname))


# encode the generated files
class encode_corpus:
    """Encode the vrt/xml files for cwb."""

    def __init__(self, mydict: dict) -> None:

        # self.corpusdir = "/home/jovyan/corpus"
        # corpusdir and regdir need to be set from input dict
        # plus we also need to set the corpus name from input dict
        self.tool = mydict["tool"]
        self.jobs = mydict["processing_type"]
        dirs_dict = mydict["advanced_options"]
        dirs_dict["output_dir"] = self.fix_path(dirs_dict["output_dir"])
        print("Writing to output dir {}".format(dirs_dict["output_dir"]))
        self.corpusdir = self.fix_path(dirs_dict["corpus_dir"])
        self.regdir = self.fix_path(dirs_dict["registry_dir"])
        self.corpusname = mydict["corpus_name"]
        self.outname = dirs_dict["output_dir"] + mydict["corpus_name"]
        print("With outname {}".format(self.outname))
        self.encodedir = self.corpusdir
        print("Found outdir {}".format(self.encodedir))

        # get attribute names
        self.attrnames = OutObject.get_names()
        # self.attrnames = self.attrnames[self.tool + "_names"]

    def _get_s_attributes(self, line: str, stags: list) -> str:
        if stags is not None:
            for tag in stags:
                print("Encoding s-attribute: {}".format(tag))
                line += "-S {} ".format(tag)
        return line

    def _get_p_attributes(self, line: str, ptags: list) -> str:
        if ptags is not None:
            for tag in ptags:
                print("Encoding p-attribute: {}".format(tag))
                line += "-P {} ".format(tag)
        return line

    def setup(self) -> bool:
        """Funtion to check wheter a corpus directory exists. If existing directory is found,
        requires input of "y" to overwrite existing files. Maybe add argument to force overwrite later?.
        If directory is not found, an empty directory is created."""

        options = "[y/n]"
        # check if corpus directory exists
        print("+++ Checking for corpus +++")
        if os.path.isdir(self.encodedir):
            # if yes, ask for overwrite
            message = "Overwrite {} and {}?".format(
                self.encodedir, self.regdir + self.corpusname
            )
            purge = self.query(message, options)
            # only overwrite if "y" to prevent accidental overwrite of data
            if purge == "y":
                print("+++ Purging old corpus +++")
                if os.path.isfile(self.regdir + self.corpusname):
                    command = "rm {}".format(self.regdir + self.corpusname)
                    print(command)
                    os.system(command)
                command = "rm -r {}".format(self.encodedir)
                print(command)
                os.system(command)
                print("+++ Purged old corpus! +++")
                os.system("mkdir {}".format(self.encodedir))
                return True
            # if no permission is granted we ask what to do
            else:
                while True:
                    cont = self.query("Continue encoding?", options)
                    if cont == "y":
                        keep = self.query("Keep old parameters?", options)
                        if keep == "y":
                            return self.setup()
                        elif keep == "n":
                            self.corpusdir = self.fix_path(
                                self.query("Please provide corpus directory path: ", "")
                            )
                            print("Set new encode directory: {}".format(self.corpusdir))
                            self.regdir = self.fix_path(
                                self.query(
                                    "Please provide registry directory path: ", ""
                                )
                            )
                            print("Set new registry directory: {}".format(self.regdir))
                            self.corpusname = self.query(
                                "Please provide corpusname: ", ""
                            )
                            print("Set new corpusname: {}".format(self.corpusname))
                            self.encodedir = self.corpusdir + self.corpusname
                            return self.setup()
                        else:
                            pass
                    elif cont == "n":
                        return False
                    else:
                        cont = self.query(
                            "Invalid input, please type 'y' or 'n'.", options
                        )

        elif not os.path.isdir(self.encodedir):
            # if the directory doesn't exist we create one
            os.system("mkdir {}".format(self.encodedir))
            print("Created directory {}.".format(self.encodedir), flush=True)
            return True

    @staticmethod
    def query(query: str, options: str) -> str:
        """Function to flush query to output and return provided input."""

        print(query, flush=True)

        return input(options)

    @staticmethod
    def fix_path(path: str) -> str:
        """Convenience function to fix provided paths to directories if neccessary."""

        if not path.endswith("/"):
            path += "/"
        # if not path.startswith("/"):
        # path = "/" + path
        return path

    def encode_vrt(self, ptags, stags):
        """Encode a new corpus into CWB from an existing vrt file."""

        line = " "
        # find out which options are to be encoded
        line = self._get_s_attributes(line, stags)
        line = self._get_p_attributes(line, ptags)
        purged = self.setup()
        if purged:
            # call the os with the encode command
            print("Encoding the corpus...")
            print("Options are:")
            command = (
                "cwb-encode -d "
                + self.encodedir
                + " -xsBC9 -c utf8 -f "
                + self.outname
                + ".vrt -R "
                + self.regdir
                + self.corpusname
                + line
            )
            print(command)
            os.system(command)
            print("Updating the registry entry...")
            print("Options are:")
            # call the os with the registry update command
            command = "cwb-makeall -r " + self.regdir + " -V " + self.corpusname
            print(command)
            os.system(command)
        elif not purged:
            return print(OSError("Error during setup, aborting..."))

    @classmethod
    def add_tags_to_corpus(cls, mydict: dict, ptags: list, stags: list):
        """Function to add tags to an already existing corpus. Should be used with output based on
        pretokenized text decoded from said corpus to assure correct alignment.

        [Args]:
                mydict[dict]: Dictionary containing the encoding information.
                ptags[list]: List containing the ptags to be used. These are checked against ptags
                            already present in the CWB corpus registry file.
                stags[list]: List containing the stags present in the corpus.
                            Only checked for the <s>...</s> structural attribute."""

        # edit the vrt file to remove the words, this could maybe be done
        # before the vrt is even written in the first place if we know
        # that we want to add to an existing corpus

        # not really happy with how this is handled right now
        #########################################################

        new = ""
        with open(cls.outname + ".vrt", "r") as vrt:
            lines = vrt.readlines()
            for line in lines:
                if not line.startswith("<"):
                    new += line.split("\t", 1)[1]
                else:
                    new += line

        with open(cls.outname + ".vrt", "w") as vrt:
            vrt.write(new)

        ##########################################################

        # check which attributes are already present in the corpus
        with open(cls.regdir + cls.corpusname, "r+") as registry:
            attributes = []
            structures = []
            for line in registry:
                if line.startswith("ATTRIBUTE"):
                    attributes.append(line.split()[1])
                if line.startswith("STRUCTURE"):
                    structures.append(line.split()[1])

            # if ptags are already present we change them to ptag_tool, if this is also
            # already present we throw an error as the annotation does already exist
            for i, ptag in enumerate(ptags):
                if ptag in attributes:
                    print("Renaming {} to {}".format(ptag, ptag + "_" + cls.tool))
                    ptags[i] = ptag + "_" + cls.tool
                    if ptags[i] in attributes:
                        raise RuntimeError(
                            "Ptag {} does already exist for this tool.".format(ptag)
                        )

            # remove existing structural tags
            for stag in stags:
                if stag in structures:
                    stags.remove(stag)

            # build the command for encoding
            line = " "
            for ptag in ptags:
                line += "-P {} ".format(ptag)
            line = cls._get_s_attributes(cls, line, stags)
            # the "-p -" removes the inbuilt "word" attribute from the encoding process
            command = (
                "cwb-encode -d"
                + cls.encodedir
                + " -xsBC9 -c utf8 -f "
                + cls.outname
                + ".vrt -p - "
                + line
            )
            print(command)
            os.system(command)

            # update the registry with the new attributes
            print("Updating the registry entry...")
            registry.seek(0, 2)
            print("Adding ptags:")
            for ptag in ptags:
                print(ptag)
                registry.write("ATTRIBUTE {}\n".format(ptag))
            for stag in stags:
                registry.write("STRUCTURE {}\n".format(stag))


class decode_corpus(encode_corpus):
    """Class to decode corpus from cwb. Inherits encode_corpus."""

    def __init__(self, mydict) -> None:
        super().__init__(mydict)

    def decode_to_file(
        self,
        directory=os.getcwd(),
        verbose=True,
        specific={"P-Attributes": [], "S-Attributes": []},
    ):
        """Function to decode a given corpus to a .out file. If the directory to write to is not
        supposed to be the current one it can be given as paramater. Location needed relative to current
        directory."""

        # set up the directories
        if not directory.endswith("/"):
            directory += "/"
        if directory != os.getcwd() + "/":
            setback = os.getcwd() + "/"
            outpath = setback + directory + self.corpusname
        else:
            setback = directory
            outpath = setback + self.corpusname
        if not os.path.isdir(outpath):
            os.system("mkdir {}".format(outpath))

        # build and execute the command line input
        p_attr = specific["P-Attributes"]
        s_attr = specific["S-Attributes"]

        # should always be the same
        base_command = "cd {} && cwb-decode ".format(self.corpusdir)

        # decide type of output, verbose or not
        if not verbose:
            base_command += "-C "

        # set registry and define corpus for cwb-decode
        base_command += "-r {} {} ".format(self.regdir, self.corpusname)

        # set up the out-pipe and return to working directory
        pipe = "> {}.out && cd {}".format(outpath, setback)

        # if there are no specified p or s attributes we just decode all
        if p_attr == [] and s_attr == []:
            command = base_command + "-ALL " + pipe

            print("Decoding corpus into directory: {}".format(outpath))
            print(command)
            os.system(command)
            print("File {}.out written in {}.".format(self.corpusname, outpath))

        # if there are specified p or s attributes we only decode these
        elif p_attr or s_attr != [] or p_attr != [] and s_attr != []:

            p_string = ""
            s_string = ""

            for p_att in p_attr:
                p_string += "-P {} ".format(p_att)

            for s_att in s_attr:
                s_string += "-S {} ".format(s_att)

            print("Decoding corpus into directory: {}".format(outpath))
            print("Grabbing p-Attributes: {}".format(p_string))
            print("Grabbing s-Attributes: {}".format(s_string))

            command = base_command + p_string + s_string + pipe
            print(command)
            os.system(command)
            print("File {}.out written in {}.".format(self.corpusname, outpath))


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
#
# p-attributes: token, lemma, POS
#
#
#
# s-attributes: sentences, NER
#
#
#
