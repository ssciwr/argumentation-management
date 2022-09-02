# the base class and utilities are contained in this module
import json
import jsonschema
import os
import importlib_resources

pkg = importlib_resources.files("nlpannotator")


class PrepareRun:
    """Class that contains all general pre-processing methods."""

    def __init__(self) -> None:
        """Since this is a namespace class, no initialization."""
        pass

    @staticmethod
    def get_cores() -> int:
        """Find out how many CPU-cores are available for current process."""
        # may need to update this to using multiprocess
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
        """Function to load a dictionary from .json.

        Args:
                name[str]: Name of .json file (without file extension)."""

        with open("{}".format(name)) as f:
            mydict = json.load(f)
        return mydict

    # load the dictionary schema and validate against
    @staticmethod
    def validate_input_dict(dict_in: dict) -> None:
        file = pkg / "data" / "input_schema.json"
        with file.open() as f:
            myschema = json.load(f)
        jsonschema.validate(instance=dict_in, schema=myschema)


# set the string to be used for undefined tags
NOT_DEF = " "


class OutObject:
    """The base output object and namespace.

    This class is inherited in the different modules;
    selected methods are overwritten/added depending on
    the requirements of the modules.
    The mapping dict "attribute_names" is used to make the
    conversion clear and not to duplicate code.
    Write the vrt/xml file."""

    def __init__(self, doc, jobs: list, start: int, style="STR"):
        self.doc = doc
        self.jobs = jobs
        self.start = start
        self.style = style
        # just one doc object for whole text or multiple objects per sentence
        # get the attribute names for the different tools
        self.attrnames = self.get_names()
        # ptags does the same case selection as OutObject.collect_results, so the order
        # of .vrt and this list should always be identical. If you change one
        # MAKE SURE to also change the other.
        self.ptags = []

    @staticmethod
    def open_outfile(outname: str):
        """Initialize output file.

        Args:
                outname[str]: Name of file to be created in out directory."""

        name = "out/" + outname
        f = open(name, "w")
        return f

    def iterate(self, out, sent):
        """Iterate through the tokens in a sentence."""
        for token in sent:
            line = token.text
            out.append(line + "\n")
        return out

    def assemble_output_sent(self) -> list:
        """Template function to assemble output for tool at sentence level."""

        # insert sentence symbol <s> before and </s> after every sentence
        if "sentence" not in self.attrnames:
            raise KeyError("Error: Sentence-Key not in obj.attrnames.")
        # leaving start in as may be needed for xml writing
        self.tstart = 0
        out = []
        for sent in getattr(self.doc, self.attrnames["sentence"]):
            out.append("<s>\n")
            out = self.iterate(out, sent)
            out.append("</s>\n")
        return out

    def assemble_output_tokens(self, out) -> list:
        """Template function to assemble output for tool at token level."""
        # this needs to be done module-specific for now and is set in each subclass
        return out

    def iterate_tokens(self, out, token_list):
        """Assemble output for tool at token level."""
        token_list_out = self.out_shortlist(out)
        # now compare the tokens in out with the tokens from the current tool
        for token_tool, token_out in zip(token_list, token_list_out):
            mylen = len(token_tool.text)
            # print("Checking for tokens {} {}".format(token_tool.text, token_out[0]))
            # check that the text is the same
            if token_tool.text != token_out[0][0:mylen]:
                raise RuntimeError(
                    "Found different token than in out! - {} and {}. Please check your inputs!".format(
                        token_tool.text, token_out[0][0:mylen]
                    )
                )
            else:
                line = self.collect_results(token_tool, 0, token_tool)
                # now replace the respective token with annotated token
                out[token_out[1]] = out[token_out[1]].replace("\n", "") + line + "\n"
                # note that this will not add a linebreak for <s> and <s\> -
                # linebreaks are handled by sentencizer
                # we expect that sentencizer runs
                # TODO be able to feed only one sentence
        return out

    def token_list(self, myobj) -> list:
        """Convert tokens from object into list."""
        return [token for token in myobj]

    def out_shortlist(self, out: list) -> list:
        """Remove the structural attributes before and after sentence to compare tokens."""
        out = [
            (token.strip(), i)
            for i, token in enumerate(out)
            if token.strip() != "<s>" and token.strip() != "</s>"
        ]
        return out

    def _compare_tokens(self, token1, token2):
        """Find out if tokens from previous and current tool are identical."""
        return token1 == token2

    @staticmethod
    def get_names() -> dict:
        """Load attribute names for specific tools."""
        file = pkg / "data" / "attribute_names.json"
        mydict = PrepareRun.load_input_dict(file)
        return mydict

    # This is currently not working properly
    # as cwb requires a semi-vrt format also for the xml
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

    def get_stags(self) -> list:
        stags = []
        if any(attr in self.attrnames["proc_sent"] for attr in self.jobs):
            stags.append("s")
        if stags == []:
            stags = None
        return stags

    def collect_results(self, token, tid: int, word) -> dict or str:

        """Function to collect requested tags for tokens after applying pipeline to data.

        Args:
                style[str]. Return line as string (STR) for .vrt or dict (DICT) for .xml."""

        # always get token id and token text
        line = {"id": str(tid), "text": token.text}
        # put in correct order - first pos, then lemma, then ner
        # order matters for encoding

        if self.attrnames["proc_pos"] in self.jobs:
            if "pos" not in self.ptags:
                self.ptags.append("pos")
            line["POS"] = self.grab_tag(word)

        if self.attrnames["proc_lemma"] in self.jobs:
            if "lemma" not in self.ptags:
                self.ptags.append("lemma")
            line["LEMMA"] = self.grab_lemma(word, self.attrnames["lemma"])

        if "ner" in self.jobs:
            if "ner" not in self.ptags:
                self.ptags.append("NER")
            line["NER"] = self.grab_ent(token)

        return self.switch_style(line)

    def grab_tag(self, word):
        """Get the pos."""
        if getattr(word, self.attrnames["pos"]) != "":
            tag = getattr(word, self.attrnames["pos"])
        else:
            tag = NOT_DEF
        return tag

    def grab_lemma(self, word, attrname):
        """Get the lemma."""
        if word.lemma != "":
            tag = getattr(word, attrname)
        else:
            tag = NOT_DEF
        return tag

    # if this is to be used: needs to be checked for
    # correct attribute names for all tools
    def grab_ent(self, token):
        """Get the named entity properties."""
        if token.ent_type_ != "":
            tag = token.ent_type_
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
    def write_xml(corpus_name: str, outname: str, out: list) -> None:
        """CWB requires a semi-vrt xml including tab spaces."""
        string = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        string += '<corpus name="{}">\n'.format(corpus_name)
        string += "<text>\n"
        for line in out:
            string += line
        string += "</text>\n"
        string += "</corpus>"
        with open("{}.xml".format(outname), "w") as file:
            file.write(string)
        print("+++ Finished writing {}.xml +++".format(outname))


# encode the generated files
# Right now we don't need this
class EncodeCorpus:
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

    # this needs refactor TODO
    def setup(self) -> bool:
        """Funtion to check wheter a corpus directory exists. If directory is not found, an empty directory is created."""

        # check if corpus directory exists
        print("+++ Checking for corpus +++")
        if os.path.isdir(self.encodedir):
            # if yes, overwrite
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
        print(stags, ptags, "have been found")
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

    # this needs refactor TODO
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


# this needs refactor TODO
class DecodeCorpus(EncodeCorpus):
    """Class to decode corpus from cwb. Inherits EncodeCorpus."""

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
