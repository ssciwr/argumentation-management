import spacy as sp
import base as be

# from spacy.lang.en import English

# from collections import defaultdict

# maybe for parallelising the pipeline, this should be done
# in base though
import os

# cores = len(os.sched_getaffinity(0))


# initialize the spacy top level class, this does
# -> define how to read from the config dict


class spacy:
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

    def __init__(self, config):

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
class spacy_pipe(spacy):
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
    def apply_to(self, data):

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data)
        return self

    def collect_results(self, token, out, start=0):
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

    def assemble_output_sent(self, start=0):

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

    def assemble_output(self, start=0):

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

    def to_vrt(self, ret=False, start=0):
        """Function to build list with results from the doc object
        and write it to a .vrt file.

        -> can only be called after pipeline was applied.

        [Args]:
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
            start[int]: Starting index for token indexing in passed data, usefull if data is chunk of larger corpus.
        """
        if "senter" in self.jobs or "sentencizer" in self.jobs or "parser" in self.jobs:
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


# maybe this could be a thing in base class as we may need something similiar for the other methods
def find_last_idx(chunk):
    """Function to find last index in output from spacy_pipe

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
    senter_config = {
        "filename": "Test1",
        "lang": "en",
        "text_type": "news",
        "processors": "tok2vec,tagger,attribute_ruler,lemmatizer",
        "pretrained": False,
        "set_device": False,
        "config": {},
    }

    # spacy_pipe(senter_config).apply_to(data).to_vrt()

    # try to chunk the plenary text from example into pieces, annotate these and than reasemble to .vrt
    def chunk_sample_text(path):
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
                    # if we are not in paragraph:
                    if inpar is False:
                        # we are now in paragraph
                        inpar = True
                        # add chunk to list-> chunk is list of three strings:
                        # chunk[0]: Opening "<>" statement
                        # chunk[1]: Text contained in chunk, every "\n" replaced with " "
                        # chunk[2]: Next "<>" statement
                        data.append(["", "", ""])
                        data[i][0] += line.replace("\n", " ")
                    # if we are in paragraph
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

    # get chunked text
    data = chunk_sample_text("data/Original/plenary.vrt")

    # start with basic config as above
    config = {
        "filename": "test",
        "processors": "tok2vec, tagger, parser,\
            attribute_ruler, lemmatizer, ner",
        "text_type": "news",
        "pretrained": "en_core_web_sm",
        "set_device": False,
        "config": {},
    }

    # load pipeline, we could also do this later, use different configs for different chunks etc.
    nlp = spacy_pipe(config)
    # output to create .vrt from
    out = []

    for i, chunk in enumerate(data):
        # get the "< >" opening statement
        out.append(data[i][0] + "\n")
        if i == 0:
            # apply pipe to chunk, token index from 0
            tmp = nlp.apply_to(chunk[1]).to_vrt(ret=True)
        elif i > 0:
            # apply pipe to chunk, keeping token index from previous chunk
            tmp = nlp.apply_to(chunk[1]).to_vrt(
                ret=True, start=find_last_idx(tmp) + 1
            )  # int(tmp[-2].split()[0]+1))
        # append data from tmp pipe output to complete output
        for line in tmp:
            out.append(line)
        # append the "< >" closing statement
        out.append(data[i][2] + "\n")

    # write complete output to file
    with open("{}_spacy.vrt".format("TestPlenary"), "w") as file:
        for chunk in out:
            for line in chunk:
                file.write(line)

    # maybe enable loading of processors from different models?

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
