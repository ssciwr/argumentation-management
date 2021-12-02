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
            # define language -> is this smart or do we want to load a model and disable?
            # -> keep for now, as I am not sure
            self.nlp = sp.blank(self.config["lang"])
            # go through the requested processors
            print("*" * 40)
            for factory in self.config["add_to"]:
                # check if the keywords requested correspond to factories in language
                if factory in self.nlp.factories:
                    print(
                        "Factory for {} found in language {}.".format(
                            factory, self.config["lang"]
                        )
                    )
                    # if yes:
                    # try to add the processor from the source model
                    # I do assume that one would add components from a model that exist
                    # within the given model -> you should know the model you're using
                    # -> If we want to check that these are valid we would have to load the model
                    # first I think
                    try:
                        self.nlp.add_pipe(
                            factory, source=sp.load(self.config["source"])
                        )
                        print(
                            "Added {} from model {}".format(
                                factory, self.config["source"]
                            )
                        )
                    # if source can't be found, ask to try and download it
                    # -> files can be big, especially the trf pipelines
                    except OSError:
                        inp = input(
                            "Could not find {} on system. Attempt to download? [Y/N]".format(
                                self.config["source"]
                            )
                        )

                        if inp == "Y":
                            os.system(
                                "python -m spacy download {}".format(
                                    self.config["source"]
                                )
                            )
                            self.nlp.add_pipe(
                                factory, source=sp.load(self.config["source"])
                            )
                        elif inp == "N":
                            exit()
                    except KeyError:
                        message = "You may have tried to add a porcessor that isn't defined in the source model.\n\
                                   \rIf you're loading a pretrained spaCy pipeline you may find a list of available keywords at:\n\
                                   \rhttps://spacy.io/models/{}#{}".format(
                            self.config["lang"], self.config["source"]
                        )
                        print(message)
                        exit()
                # if no:
                else:
                    # can't add processor as there are no instructions for initialization
                    # input cannot be executed
                    # one may be abled to define custom factories by adding them to the languages config file
                    raise KeyError(
                        "Requested Processor {} doesn't correspond to a factory in {}.".format(
                            factory, self.config["lang"]
                        )
                    )
            print("*" * 40)

    # call the build pipeline on the data
    def apply_to(self, data):
        self.doc = self.nlp(data)
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
            "lang": "en",
            "add_to": [
                "tok2vec",
                "senter",
                "tagger",
                "parser",
                "attribute_ruler",
                "lemmatizer",
                "ner",
            ],
            "source": "en_core_web_sm",
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
            "lang": "en",
            "add_to": ["tok2vec", "attribute_ruler", "lemmatizer"],
            "source": "en_core_web_md",
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
