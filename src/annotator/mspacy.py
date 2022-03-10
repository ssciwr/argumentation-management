import spacy as sp
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from spacy.lang.de import German
import copy
import base as be
from tqdm import (
    tqdm,
)  # for progress in pipe_multiple, might be interesting for large corpora down the line


available_lang = ["en", "de"]


class Spacy:
    """Base class for spaCy module.

    Args:
        config[dict]: Dict containing the setup for the spaCy run.
    """

    def __init__(self, config: dict):

        # config = the input dictionary
        # output file name
        config = be.prepare_run.update_dict(config)
        # check for pretrained
        # lets you initialize your models with information from raw text
        # you would do this if you had generated the model yourself
        self.pretrained = config["pretrained"]

        if self.pretrained:
            self.model = self.pretrained

        # here we put some sensible default values
        # in general, it should also be possible
        # that the user puts in a language and selected model
        # so if model is already set, we should not overwrite it
        # I believe this would then eliminate the `elif` case for pretrained above
        self.lang = config["lang"]
        self.type = config["text_type"]
        if "model" in config and config["model"] is not False:
            self.model = config["model"]
            print("Using selected model {}.".format(self.model))
        else:
            # now here goes the default model if none was selected
            if self.lang == "en":
                if self.type == "news":
                    self.model = "en_core_web_md"

            elif self.lang == "de":
                if self.type == "news":
                    self.model = "de_core_news_md"
                elif self.type == "biomed":
                    # uses the scispacy package for processing biomedical text
                    self.model = "en_core_sci_md"
            # make sure to throw an exception if language is not found
            # the available languages should be stored in a list somewhere
            # put it on top of the module for now, find a better place for it later.
            else:
                raise ValueError(
                    """Languages not available yet. Only {} models have been implemented.
                Aborting...""".format(
                        available_lang
                    )
                )

        # get processors from dict
        # self.jobs = config["processors"]
        self.jobs = be.prepare_run.get_jobs(config)

        # use specific device settings if requested
        if config["set_device"]:
            if config["set_device"] == "prefer_GPU":
                sp.prefer_gpu()
            elif config["require_GPU"] == "require_GPU":
                sp.require_gpu()
            elif config["require_CPU"] == "require_CPU":
                sp.require_cpu()

        self.config = config["config"]
        self.config = be.prepare_run.update_dict(self.config)


# build the pipeline from config-dict
class spacy_pipe(Spacy):
    """Assemble pipeline from config, apply pipeline to data and write results to .vrt file."""

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
                raise OSError(
                    "Could not find {} in standard directory.".format(self.model)
                )

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
                    raise ValueError(message)
            print(">>>")

            # assemble list of excluded components from list of available components and
            # validated list of existing components so not to load in the pipeline
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

        Args:
                chunks[list[list[str]]]: List of chunks which are lists containing
                                            [opening <>, text, closing <>].

                ret[bool]=False: Wheter to return output as list (True) or write to file (False).
        """

        out = []

        # apply pipe to list of text chunks and write resulting
        # generator to list of docs
        text = [List[1] for List in chunks]
        n_process = be.prepare_run.get_cores()
        self.docs = list(
            tqdm(
                self.nlp.pipe(text, n_process=n_process),
                total=len(text),
                desc="Running on {} cores".format(n_process),
                bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                unit="chunks",
            )
        )
        #         self.docs = list(self.nlp.pipe(text, n_process=be.prepare_run.get_cores()))
        # iterate through doc objects
        for i, doc in enumerate(self.docs):
            # get the "< >" opening statement
            out.append(chunks[i][0] + "\n")
            # self.doc is now current doc
            self.doc = doc
            tmp = self.pass_results(ret=True, start=0)
            # append data from tmp output to complete output
            for line in tmp:
                out.append(line)
            # append the "< >" closing statement
            out.append(chunks[i][2] + "\n")

        if ret is False:
            be.out_object.write_vrt(self.outname, out)

        else:
            return out

    def pass_results(self, out_param=None, ret=False, start=0) -> list or None:
        """Function to build list with results from the doc object
        and write it to a .vrt file / encode to cwb directly.

        -> can only be called after pipeline was applied.

        Args:
            out_param[dict]: Dict containing the information to encode the .vrt for cwb.
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
            start[int]: Starting index for token indexing in passed data, useful if data is chunk of larger corpus.
        """

        out = out_object_spacy(self.doc, self.jobs, start=start).fetch_output()
        # write to file -> This overwrites any existing file of given name;
        # as all of this should be handled internally and the files are only
        # temporary, this should not be a problem. right?
        if ret is False and out_param is not None:
            be.out_object.write_vrt(out_param["output"], out)
            # encode
            be.encode_corpus.encode_vrt(out_param)

        elif ret is True:
            return out

        else:
            raise RuntimeError(
                "If ret is not set to True, a dict containing the encoding parameters is needed!"
            )

    def get_multiple(self, chunks: list, ret=False) -> list or None:
        """Iterate through a list of chunks generated by be.chunk_sample_text, tag the tokens
        and create output for full corpus, either return as list or write to .vrt. This function is
        essentially equal to, but slower than, spacy_pipe.pipe_multiple.

        Args:
                chunks[list[list[str]]]: List of chunks which are lists containing
                                            [opening <>, text, closing <>].

                ret[bool]=False: Wheter to return output as list (True) or write to file (False).
        """

        out = []

        for i, chunk in enumerate(chunks):
            # get the "< >" opening statement
            out.append(chunks[i][0] + "\n")
            # apply pipe to chunk, token index from 0
            tmp = self.apply_to(chunk[1]).pass_results(ret=True)
            # append data from tmp pipe output to complete output
            for line in tmp:
                out.append(line)
            # append the "< >" closing statement
            out.append(chunks[i][2] + "\n")

        if ret:
            return out

        elif not ret:
            # write complete output to file
            with open("{}_spacy.vrt".format(self.outname), "w") as file:
                for chunk in out:
                    for line in chunk:
                        file.write(line)
                print("+++ Finished writing {}.vrt +++".format(self.outname))


def sentencize_spacy(lang: str, data: str) -> list:
    """Function to sentencize given text data in either German or English language.

    Args:
            data[str]: The text string to be split into sentences.

    Returns:
            List[List[str, int]]: List containing lists which contain the sentences as strings
            as well as the number of tokens previous to the sentence to easily keep
            track of the correct token index for a given sentence in the list.
    """

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


# inherit the output class from base and add spacy-specific methods
class out_object_spacy(be.out_object):
    """Out object for spacy annotation, adds spacy-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs, start):
        super().__init__(doc, jobs, start)
        self.attrnames = self.attrnames["spacy_names"]

    def iterate(self, out, sent):
        for token in sent:
            # multi-word expressions not available in spacy?
            # Setting word=token for now
            tid = copy.copy(token.i)
            out, line = self.collect_results(token, tid, token, out)
            out.append(line + "\n")
        return out

    def fetch_output(self) -> list:
        """Function to assemble the output list for a run. Can work with or without sentence
        level annotation and will check if doc is sentencized on its own."""

        try:
            assert hasattr(self, "doc")
        except AttributeError:
            print(
                "Seems there is no Doc object, did you forget to call spaCy_pipe.apply_to()?"
            )
            exit()

        out = []
        # check if spacy doc object is sentencized
        if self.doc.has_annotation("SENT_START"):
            # apply sentence and sublevel annotation
            out = self.assemble_output_sent(self.doc, self.jobs, start=self.start)

        # if not sentencized just iterate doc and extract the results
        elif not self.doc.has_annotation("SENT_START"):
            out = self.iterate(out, self.doc)
        return out
