import spacy as sp
from spacy.lang.en import English

# from collections import defaultdict

# maybe for parallelising the pipeline, this should be done
# in base though
import os

# cores = len(os.sched_getaffinity(0))


# initialize the spacy top level class, this does
# -> define how to read from the config dict


class spaCy:
    """Base class for spaCy module

    Args:
        config[dict]: Dict containing the setup for the spaCy run.
            -> structure:
                config = {
                    "JobID": "str",
                    "config": {"model":str, "add_to":list(str)},
                    "pretrained"=bool
                    }

                JobID: String with ID to be used for saving to .vrt file
                config: dict holding information to assemble the pipeline
                        -> model to load, which components to add
                pretrained: Use specific pipeline with given name/from given path
    """

    def __init__(self, config):

        # config to build a pipeline
        self.JobID = config["JobID"]
        self.config = config["config"]
        self.jobs = config["config"]["add_to"]
        #  list of which Results to fetch, ideally corresponding
        # to loaded components in config, currently only supports
        # "NER"
        # use a full pretrained pipeline if desired, this will run
        # all components of said pipeline
        if config["pretrained"]:
            self.pretrained = config["pretrained"]
        else:
            self.pretrained = False


# build the pipeline from config-dict
class spaCy_pipe(spaCy):
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
    def __init__(self, config):
        super().__init__(config)

        # use a specific pipeline if requested
        if self.pretrained:
            self.nlp = sp.load(self.pretrained)

        # initialize pipeline
        else:
            self.used = []
            # define language -> is this smart or do we want to load a model and disable?
            # -> changed it to load a model and disable, as I was experiencing inconsistencies
            # with building from base language even for just the two models I tried
            try:
                self.nlp = sp.load(self.config["model"])
            # if model can't be found, ask to try and download it
            # -> files can be big, especially the trf pipelines
            except OSError:
                inp = input(
                    "Could not find {} on system. Attempt to download? [Y/N]".format(
                        self.config["model"]
                    )
                )

                if inp == "Y":
                    os.system(
                        "python -m spacy download {}".format(self.config["model"])
                    )
                elif inp == "N":
                    exit()

            print("*" * 50)

            # find which processors are available in model
            components = [component[0] for component in self.nlp.components]

            # go through the requested processors
            for component in self.config["add_to"]:
                # check if the keywords requested correspond to available components in pipeline
                if component in components:
                    # if yes:
                    print(
                        "Loading component {} from model {}.".format(
                            component, self.config["model"]
                        )
                    )
                    # add to list of components to be used
                    self.used.append(component)

                # if no, there is maybe a typo, display some info and try to link to spacy webpage of model
                # -> links may not work if they change their websites structure in the future
                else:
                    print(
                        "Component {} not found in model {}.".format(
                            component, self.config["model"]
                        )
                    )
                    message = "You may have tried to add a processor that isn't defined in the source model.\n\
                            \rIf you're loading a pretrained spaCy pipeline you may find a list of available keywords at:\n\
                            \rhttps://spacy.io/models/{}#{}".format(
                        "{}".format(self.config["model"].split("_")[0]),
                        self.config["model"],
                    )
                    print(message)
                    exit()
            print("*" * 50)

    # call the build pipeline on the data
    def apply_to(self, data):

        # create an empty list for the disabled components
        disable = []
        # fill it up with everything that isn't in the used list
        # -> used list should only contain validated components at this point
        for tupple in self.nlp.components:
            if tupple[0] not in self.used:
                disable.append(tupple[0])

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data, disable=disable)
        return self

    def assemble_output_sent(self):

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
                # always get token id and token text
                line = str(token.i) + " " + token.text

                # grab the data for the run components, I've only included the human readable
                # part of output right now as I don't know what else we need
                if "ner" in self.jobs:
                    if token.i == 0:
                        out[1] += " ner"
                    if token.ent_type_ != "":
                        line += "  " + token.ent_type_
                    else:
                        line += " - "

                if "lemmatizer" in self.jobs:
                    if token.i == 0:
                        out[1] += " lemma"
                    line += " " + token.lemma_

                if "tagger" in self.jobs:
                    if token.i == 0:
                        out[1] += " Tag"
                    line += " " + token.tag_

                if "parser" in self.jobs:
                    if token.i == 0:
                        out[1] += " Depend"
                    line += " " + token.dep_

                if "attribute_ruler" in self.jobs:
                    if token.i == 0:
                        out[1] += " POS"
                    line += " " + token.pos_
                    # add what else we need

                out.append(line + "\n")
            out.append("</s>\n")
        out[1] += " \n"
        return out

    def assemble_output(self):

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
            line = str(token.i) + " " + token.text
            if "ner" in self.jobs:
                if token.i == 0:
                    out[1] += " ner"
                if token.ent_type_ != "":
                    line += "  " + token.ent_type_
                else:
                    line += " - "

            if "lemmatizer" in self.jobs:
                if token.i == 0:
                    out[1] += " lemma"
                line += "  " + token.lemma_

            if "tagger" in self.jobs:
                if token.i == 0:
                    out[1] += " Tag"
                line += "  " + token.tag_

            if "parser" in self.jobs:
                if token.i == 0:
                    out[1] += " Depend"
                line += "  " + token.dep_

            if "attribute_ruler" in self.jobs:
                if token.i == 0:
                    out[1] += " POS"
                line += "  " + token.pos_
                out.append(line + "\n")
        out[1] += " \n"

        return out

    def to_vrt(self):
        """Function to build list with results from the doc object
        and write it to a .vrt file.

        -> can only be called after pipeline was applied.
        """
        if "senter" in self.jobs:
            out = self.assemble_output_sent()
        else:
            out = self.assemble_output()
        # write to file -> This overwrites any existing file of given name
        # as all of this should be handled internally and the files are only
        # temporary, this should not be a problem. right?
        with open("{}_spacy.vrt".format(self.JobID), "w") as file:
            for line in out:
                file.write(line)


if __name__ == "__main__":
    with open("../../data/Original/iued_test_original.txt", "r") as file:
        data = file.read().replace("\n", "")

    # lets emulate a run of en_core_web_sm
    config = {
        "JobID": "Test",
        "config": {
            "model": "en_core_web_sm",
            "add_to": [
                "tok2vec",
                "senter",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
        },
        # why distinguish jobs and processors? -> The processors are not neccessarily named uniformly for
        # all models, but I'm not entirely happy with this. Don't have a better idea though, except drop
        # the concept completely and come up with something new -> Think on this...
        "pretrained": False,
    }

    # build pipe from config, apply it to data, write results to vrt
    spaCy_pipe(config).apply_to(data).to_vrt()

    # this throws a warning that the senter may not work as intended, it seems to work
    # fine though
    senter_config = {
        "JobID": "Test1",
        "config": {
            "model": "en_core_web_md",
            "add_to": ["tok2vec", "tagger", "attribute_ruler", "lemmatizer"],
        },
        "pretrained": False,
    }

    spaCy_pipe(senter_config).apply_to(data).to_vrt()

    # maybe enable loading of processors from different models?

with open("Test_spacy.vrt", "r") as file:
    for line in file:
        if line != "<s>\n" and line != "</s>\n":
            try:
                assert len(line.split()) == len(config["config"]["add_to"])
            except AssertionError:
                print(line)
