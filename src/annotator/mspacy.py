import spacy as sp
from spacy.tokens.doc import Doc
from spacy.tokens.token import Token
from spacy.lang.en import English
from spacy.lang.de import German
import base as be

# import timeit
# import time

# initialize the spacy top level class, this does
# -> define how to read from the config dict


class Spacy:
    """Base class for spaCy module

    Args:
        config[dict]: Dict containing the setup for the spaCy run.
            -> structure:
                {
                "filename": str,
                "lang":str,
                "processors": str,
                "pretrained"= str or False
                }

                filename: String with ID to be used for saving to .vrt file.
                model: String with name of model installed in default
                    spacy directory or path to model.
                processors: Comma-separated string containing the processors
                    to be used in pipeline.
                pretrained: Use specific pipeline with given name/from given path.
    """

    def __init__(self, config: dict):

        # config = the input dictionary
        # output file name
        self.JobID = config["filename"]
        # check for pretrained
        self.pretrained = config["pretrained"]

        if self.pretrained:
            self.model = self.pretrained

        elif not self.pretrained:

            self.lang = config["lang"]
            self.type = config["text_type"]

            if self.lang == "en":
                if self.type == "news":
                    self.model = "en_core_web_md"

            elif self.lang == "de":
                if self.type == "news":
                    self.model = "de_core_news_md"
                elif self.type == "biomed":
                    # uses the scispacy package for processing biomedical text
                    self.model = "en_core_sci_md"

        # get processors from dict
        procs = config["processors"]
        # strip out blank spaces and separate processors into list
        self.jobs = [proc.strip() for proc in procs.split(",")]

        # use specific device settings if requested
        if config["set_device"]:
            if config["set_device"] == "prefer_GPU":
                sp.prefer_gpu()
            elif config["require_GPU"] == "require_GPU":
                sp.require_gpu()
            elif config["require_CPU"] == "require_CPU":
                sp.require_cpu()

        self.config = config["config"]
        self.config = be.update_dict(self.config)


# build the pipeline from config-dict
class spacy_pipe(Spacy):
    """Pipeline class for spaCy module -> inherits setup from base class

    Assemble pipeline from config, apply pipeline to data and write results to .vrt file.

    Methods:
            apply_to(data):
                apply pipeline of object to given data

            to_vrt():
                write results after applying pipeline to .vrt file
    """

    # init with specified config, this may be changed later?
    # -> Right now needs quite specific instuctions
    def __init__(self, config: dict):
        super().__init__(config)

        # use a specific pipeline if requested
        if self.pretrained:
            # load pipeline
            print("Loading full pipeline {}.".format(self.pretrained))

            self.nlp = sp.load(self.pretrained)

        # initialize pipeline
        else:
            self.validated = []
            # define language -> is this smart or do we want to load a model and disable?
            # -> changed it to load a model and disable, as I was experiencing inconsistencies
            # with building from base language even for just the two models I tried
            try:
                if self.config:
                    self.nlp = sp.load(self.model, config=self.config)
                else:
                    self.nlp = sp.load(self.model)

            except OSError:
                print("Could not find {} on system.".format(self.model))

            print(">>>")

            # find which processors are available in model
            components = [component[0] for component in self.nlp.components]

            # go through the requested processors
            for component in self.jobs:
                # check if the keywords requested correspond to available components in pipeline
                if component in components:
                    # if yes:
                    print("Loading component {} from {}.".format(component, self.model))
                    # add to list of validated components
                    self.validated.append(component)

                # if no, there is maybe a typo, display some info and try to link to spacy webpage of model
                # -> links may not work if they change their websites structure in the future
                else:
                    print(
                        "Component '{}' not found in {}.".format(component, self.model)
                    )
                    message = "You may have tried to add a processor that isn't defined in the source model.\n\
                            \rIf you're loading a pretrained spaCy pipeline you may find a list of available keywords at:\n\
                            \rhttps://spacy.io/models/{}#{}".format(
                        "{}".format(self.model.split("_")[0]),
                        self.model,
                    )
                    print(message)
                    exit()
            print(">>>")

            # assemble list of excluded components from list of available components and
            # validated list of existing components
            self.exclude = [
                component for component in components if component not in self.validated
            ]

            self.cfg = {
                "name": self.model,
                "exclude": self.exclude,
                "config": self.config,
            }

            self.nlp = sp.load(**self.cfg)

    # call the build pipeline on the data
    def apply_to(self, data: str) -> Doc:
        """Apply the objects pipeline to a given data string."""

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data)
        return self

    # or apply to chunked data
    def pipe_multiple(self, chunks: list, ret=False) -> list or None:
        """Iterate through a list of chunks generated by be.chunk_sample_text, tag the tokens
        and create output for full corpus, either return as list or write to .vrt. Faster than
        the spacy_pipe.get_multiple method by streaming chunks through spacy.pipe and
        using as many cores as available to multiprocess rather than iterating separately.

        [Args]:
                chunks[list[list[str,str,str]]]: List of chunks which are lists containing
                    [opening <>, text, closing <>].
                ret[bool]=False: Wheter to return output as list (True) or write to file (False)."""

        out = []

        # apply pipe to list of text chunks and write resulting
        # generator to list of docs
        text = [List[1] for List in chunks]
        self.docs = list(self.nlp.pipe(text, n_process=be.get_cores()))
        # iterate through doc objects
        for i, doc in enumerate(self.docs):
            # get the "< >" opening statement
            out.append(chunks[i][0] + "\n")
            # self.doc is now current doc
            self.doc = doc
            if i == 0:
                # token index from 0
                tmp = self.to_vrt(ret=True)
            if i > 0:
                # keep token index from previous chunk
                tmp = self.to_vrt(ret=True, start=be.find_last_idx(tmp) + 1)
            # append data from tmp output to complete output
            for line in tmp:
                out.append(line)
            # append the "< >" closinmg statement
            out.append(chunks[i][2] + "\n")

        if ret:
            return out

        elif not ret:
            # write complete output to file
            with open("{}_spacy.vrt".format(self.JobID), "w") as file:
                for chunk in out:
                    for line in chunk:
                        file.write(line)
                print("+++ Finished writing .vrt +++")

    def collect_results(self, token: Token, out: list, start=0) -> tuple:
        """Function to collect requested tags for tokens after applying pipeline to data."""
        # always get token id and token text
        line = str(token.i + start) + " " + token.text

        # grab the data for the run components, I've only included the human readable
        # part of output right now as I don't know what else we need
        if "ner" in self.jobs:
            out, line = be.out_object.grab_ner(token, out, line)

        if "entity_ruler" in self.jobs:
            out, line = be.out_object.grab_ruler(token, out, line)

        if "entity_linker" in self.jobs:
            out, line = be.out_object.grab_linker(token, out, line)

        if "lemmatizer" in self.jobs:
            out, line = be.out_object.grab_lemma(token, out, line)

        if "morphologizer" in self.jobs:
            out, line = be.out_object.grab_morph(token, out, line)

        if "tagger" in self.jobs:
            out, line = be.out_object.grab_tag(token, out, line)

        if "parser" in self.jobs:
            out, line = be.out_object.grab_dep(token, out, line)

        if "attribute_ruler" in self.jobs:
            out, line = be.out_object.grab_att(token, out, line)
            # add what else we need

        return out, line

    def assemble_output_sent(self, start=0) -> list:
        """Function to assemble the output list for a run with sentence level annotation."""

        try:
            assert self.doc
        except AttributeError:
            print(
                "Seems there is no Doc object, did you forget to call spaCy_pipe.apply_to()?"
            )
            exit()
        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence -> Is this the right symbol?
        out = ["! spaCy output for {}! \n".format(self.JobID)]
        out.append("! Idx Text")

        for sent in self.doc.sents:
            out.append("<s>\n")
            # iterate through the tokens of the sentence, this is just a slice of
            # the full doc
            for token in sent:
                out, line = self.collect_results(token, out, start=start)
                out.append(line + "\n")

            out.append("</s>\n")
        out[1] += " \n"
        return out

    def assemble_output(self, start=0) -> list:
        """Funtion to assemble the output list for a run below sentence level."""

        try:
            assert self.doc
        except AttributeError:
            print(
                "Seems there is no Doc object, did you forget to call spaCy_pipe.apply_to()?"
            )
            exit()
        # if no senter was called we either dont want to distinguish sentences
        # or passed data below sentence level -> only work on individual tokens
        out = ["! spaCy output for {}! \n".format(self.JobID)]
        out.append("! Idx Text")

        for token in self.doc:
            out, line = self.collect_results(token, out, start=start)
            out.append(line + "\n")

        out[1] += " \n"

        return out

    def to_vrt(self, ret=False, start=0) -> list or None:
        """Function to build list with results from the doc object
        and write it to a .vrt file.

        -> can only be called after pipeline was applied.

        [Args]:
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
            start[int]: Starting index for token indexing in passed data, usefull if data is chunk of larger corpus.
        """

        if self.doc.has_annotation("SENT_START"):
            # if "senter" in self.jobs or "sentencizer" in self.jobs or "parser" in self.jobs:
            out = self.assemble_output_sent(start=start)
        else:
            out = self.assemble_output(start=start)
        # write to file -> This overwrites any existing file of given name;
        # as all of this should be handled internally and the files are only
        # temporary, this should not be a problem. right?
        if ret is False:
            with open("{}_spacy.vrt".format(self.JobID), "w") as file:
                for line in out:
                    file.write(line)
            print("+++ Finished writing .vrt +++")
        else:
            return out

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def get_multiple(self, chunks: list, ret=False) -> list or None:
        """Iterate through a list of chunks generated by be.chunk_sample_text, tag the tokens
        and create output for full corpus, either return as list or write to .vrt. This function is
        essentially equal to, but slower than, spacy_pipe.pipe_multiple.

        [Args]:
                chunks[list[list[str,str,str]]]: List of chunks which are lists containing
                    [opening <>, text, closing <>].
                ret[bool]=False: Wheter to return output as list (True) or write to file (False)."""

        out = []

        for i, chunk in enumerate(chunks):
            # get the "< >" opening statement
            out.append(chunks[i][0] + "\n")
            if i == 0:
                # apply pipe to chunk, token index from 0
                tmp = self.apply_to(chunk[1]).to_vrt(ret=True)
            elif i > 0:
                # apply pipe to chunk, keeping token index from previous chunk
                tmp = self.apply_to(chunk[1]).to_vrt(
                    ret=True, start=be.find_last_idx(tmp) + 1
                )  # int(tmp[-2].split()[0]+1))
            # append data from tmp pipe output to complete output
            for line in tmp:
                out.append(line)
            # append the "< >" closing statement
            out.append(chunks[i][2] + "\n")

        if ret:
            return out

        elif not ret:
            # write complete output to file
            with open("{}_spacy.vrt".format(self.JobID), "w") as file:
                for chunk in out:
                    for line in chunk:
                        file.write(line)
                print("+++ Finished writing .vrt +++")

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


def sentencize_spacy(lang: str, data: str) -> list:
    """Function to sentencize given text data in either German or English language.

    [args]:
            data[str]: The text string to be split into sentences.

    [returns]:
            List[List[str, int]]: List containing lists which contain the sentences as strings
                as well as the number of tokens previous to the sentence to easily keep
                track of the correct token index for a given sentence in the list."""

    if lang == "en":
        nlp = English()
        nlp.add_pipe("sentencizer")

    elif lang == "de":
        nlp = German()
        nlp.add_pipe("sentencizer")

    doc = nlp(data)
    assert doc.has_annotation("SENT_START")

    sents = []

    for i, sent in enumerate(doc.sents):
        if i == 0:
            # need to take the len of the split str as otherwise grouping of multiple tokens by
            # spacy can be a problem. This now assumes that tokens are always separated by a
            # whitespace, which seems reasonable to me -> Any examples to the contrary?
            sents.append([sent.text, len(sent.text.split())])
        elif i > 0:
            sents.append([sent.text, len(sent.text.split()) + sents[i - 1][1]])
    return sents


if __name__ == "__main__":
    data = be.get_sample_text()
    # lets emulate a run of en_core_web_sm
    # sample dict -> keep this structure or change to structure from spacy_test.ipynb?
    config = {
        "filename": "Test",
        "lang": "en",
        "text_type": "news",
        "processors": "tok2vec, tagger, parser,\
            attribute_ruler, lemmatizer, ner",
        "pretrained": False,
        "set_device": False,
        "config": {
            "nlp.batch_size": 512,
            "components": {
                "attribute_ruler": {"validate": True},
                "lemmatizer": {"mode": "rule"},
            },
        },
    }
    # or read the main dict and activate
    mydict = be.load_input_dict()
    # take only the part of dict pertaining to spacy
    # filename needs to be moved to/taken from top level of dict
    spacy_dict = mydict["spacy_dict"]
    # remove comment lines starting with "_"
    spacy_dict = be.update_dict(spacy_dict)
    # build pipe from config, apply it to data, write results to vrt
    spacy_pipe(spacy_dict).apply_to(data).to_vrt()
    # spacy_pipe(config).apply_to(data).to_vrt()

    # this throws a warning that the senter may not work as intended, it seems to work
    # fine though
    # senter_config = {
    #     "filename": "Test1",
    #     "lang": "en",
    #     "text_type": "news",
    #     "processors": "tok2vec,tagger,attribute_ruler,lemmatizer",
    #     "pretrained": False,
    #     "set_device": False,
    #     "config": {},
    # # }

    # # spacy_pipe(senter_config).apply_to(data).to_vrt()
    # # try to chunk the plenary text from example into pieces, annotate these and than reasemble to .vrt
    # # get chunked text
    # data = be.chunk_sample_text("data/Original/plenary.vrt")

    # # start with basic config as above if we use the pretrained keyword it
    # # replaces the lang and text_type keys so we don't need to specifiy them

    # # some testing to check wheter spacy_pipe.pipe_multiple or spacy_pipe.get_multiple is faster

    # config = {
    #     "filename": "test_new",
    #     "processors": "tok2vec, tagger, parser,\
    #         attribute_ruler, lemmatizer, ner",
    #     "pretrained": "de_core_news_md",
    #     "set_device": False,
    #     "config": {"nlp.batch_size": 10},
    # }

    # config1 = {
    #     "filename": "test_class",
    #     "processors": "tok2vec, tagger, parser,\
    #         attribute_ruler, lemmatizer, ner",
    #     "pretrained": "de_core_news_md",
    #     "set_device": False,
    #     "config": {"nlp.batch_size": 10},
    # }

    # repeat = 50

    # pipe = spacy_pipe(config)

    # stmt = """pipe.pipe_multiple(data)"""

    # Time = timeit.repeat(
    #     stmt=stmt, number=1, repeat=repeat, timer=time.process_time, globals=globals()
    # )

    # check = []
    # with open("test_new_spacy.vrt", "r") as file:
    #     for line in file:
    #         if not line.startswith("!") and not line.startswith("<"):
    #             check.append(int(line.split()[0]))

    # for i, elem in enumerate(check):
    #     if i > 0:
    #         try:
    #             assert elem - check[i - 1] == 1
    #         except AssertionError:
    #             print(i, elem)
    # print("Asserted pipe indexing.")

    # pipe = spacy_pipe(config1)

    # stmt1 = """pipe.get_multiple(data)"""

    # Time1 = timeit.repeat(
    #     stmt=stmt1, number=1, repeat=repeat, timer=time.process_time, globals=globals()
    # )

    # check = []
    # with open("test_class_spacy.vrt", "r") as file:
    #     for line in file:
    #         if not line.startswith("!") and not line.startswith("<"):
    #             check.append(int(line.split()[0]))

    # for i, elem in enumerate(check):
    #     if i > 0:
    #         try:
    #             assert elem - check[i - 1] == 1
    #         except AssertionError:
    #             print(i, elem)
    # print("Asserted iterating indexing.")

    # print(
    #     "Using spacy.pipe: {:.2f}s | Iterating directly: {:.2f}s".format(
    #         min(Time), min(Time1)
    #     )
    # )

    # # check that output is indeed the same for both methods
    # with open("test_new_spacy.vrt", "r") as file:
    #     out1 = file.readlines()

    # with open("test_class_spacy.vrt", "r") as file:
    #     out = file.readlines()

    # i = 0
    # equal = True
    # for i, line in enumerate(out):
    #     if not line.startswith("!"):
    #         try:
    #             assert line == out1[i]
    #         except AssertionError:
    #             if i < 10:
    #                 print(line, out1[i])
    #                 equal = False
    #                 i += 1
    # print("Asserted equality")


# with open("out/test_spacy.vrt", "r") as file:
# for line in file:
# check if vrt file was written correctly
# lines with "!" are comments, <s> and </s> mark beginning and
# end of sentence, respectively
# if line != "<s>\n" and line != "</s>\n" and line.split()[0] != "!":
# try:
#    assert len(line.split()) == len(spacy_dict["processors"].split(","))
# except AssertionError:
#    print(line)
