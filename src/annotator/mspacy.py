import spacy as sp
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from spacy.lang.de import German
import copy
import base as be
from tqdm import (
    tqdm,
)  # for progress in pipe_multiple, might be interesting for large corpora down the line


class MySpacy:
    """Base class for spaCy module.

    Args:
        config[dict]: Dict containing the setup for the spaCy run.
    """

    def __init__(self, config: dict):
        self.lang = config["lang"]
        self.type = config["text_type"]
        if "model" in config and config["model"] is not False:
            self.model = config["model"]
            print("Using selected model {}.".format(self.model))
        else:
            # now here goes the default model if none was selected
            # this all to be moved to base or another spot where the pipeline
            # is set
            if self.lang == "en":
                if self.type == "news":
                    self.model = "en_core_web_md"
                elif self.type == "biomed":
                    # uses the scispacy package for processing biomedical text
                    self.model = "en_core_sci_md"

            elif self.lang == "de":
                if self.type == "news":
                    self.model = "de_core_news_md"

            # make sure to throw an exception if language is not found
            # the available languages should be stored in a list somewhere
            # put it on top of the module for now, find a better place for it later.
            else:
                raise ValueError("""Languages not available yet. Aborting...""")

        self.jobs = be.prepare_run.get_jobs(config)

        # if we ask for lemma and/or POS we force tok2vec to boost accuracy
        if (
            "lemmatizer" in self.jobs
            or "tagger" in self.jobs
            or "lemmatizer"
            and "tagger" in self.jobs
        ):
            if "tok2vec" not in self.jobs:
                self.jobs = ["tok2vec"] + self.jobs

        # use specific device settings if requested
        # this also to be set in the pipeline decision
        if config["set_device"]:
            if config["set_device"] == "prefer_GPU":
                sp.prefer_gpu()
            elif config["set_device"] == "require_GPU":
                sp.require_gpu()
            elif config["set_device"] == "require_CPU":
                sp.require_cpu()

        self.config = config["config"]


# build the pipeline from config-dict
class spacy_pipe(MySpacy):
    """Assemble pipeline from config, apply pipeline to data and write results to .vrt file."""

    # init with specified config, this may be changed later?
    # -> Right now needs quite specific instuctions
    def __init__(self, config: dict):
        super().__init__(config)
        # use a specific pipeline if requested
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
            raise OSError("Could not find {} in standard directory.".format(self.model))

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
                print("Component '{}' not found in {}.".format(component, self.model))
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
            tmp = self.pass_results(style="STR", ret=True, start=0)
            # append data from tmp output to complete output
            for line in tmp:
                out.append(line)
            # append the "< >" closing statement
            out.append(chunks[i][2] + "\n")

        if ret is False:
            be.out_object.write_vrt(self.outname, out)

        else:
            return out

    def pass_results(
        self,
        mydict: dict or None = None,
        style: str = "STR",
        ret: bool = False,
        start: int = 0,
        add: bool = False,
    ) -> list or None:

        """Function to build list with results from the doc object
        and write it to a .vrt file / encode to cwb directly.

        -> can only be called after pipeline was applied.

        Args:
            mydict[dict]: Dict containing the information to encode the .vrt for cwb.
            ret[bool]: Wheter to return output as list (True) or write to .vrt file (False, Default)
            start[int]: Starting index for token indexing in passed data, useful if data is chunk of larger corpus.
            add[bool]: Indicates if a new corpus should be started or if tags should be added to existing corpus.
        """

        out_obj = out_object_spacy(self.doc, self.jobs, start=start)
        out = out_obj.fetch_output(style)
        ptags = out_obj.ptags
        stags = out_obj.stags
        # write to file -> This overwrites any existing file of given name;
        # as all of this should be handled internally and the files are only
        # temporary, this should not be a problem. right?
        if mydict is not None:
            outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
        if ret is False and style == "STR" and mydict is not None and add is False:
            be.out_object.write_vrt(outfile, out)
            # encode
            be.encode_corpus.encode_vrt(mydict, ptags, stags)

        elif ret is False and style == "STR" and mydict is not None and add is True:
            be.out_object.write_vrt(outfile, out)
            be.encode_corpus.add_tags_to_corpus(mydict, ptags, stags)

        elif ret is False and style == "DICT" and mydict is not None:
            be.out_object.write_xml(
                mydict["output"].replace("/", "_"), mydict["output"], out
            )

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
            tmp = self.apply_to(chunk[1]).pass_results(style="STR", ret=True)

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
        self.ptags = self.get_ptags()
        self.stags = self.get_stags()

    def iterate(self, out, sent, style):
        for token in sent:
            # multi-word expressions not available in spacy?
            # Setting word=token for now
            tid = copy.copy(token.i)
            line = self.collect_results(token, tid, token, style)
            if style == "STR":
                out.append(line + "\n")
            elif style == "DICT":
                out.append(line)
        return out

    def fetch_output(self, style) -> list:
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
            if style == "STR":
                out = self.assemble_output_sent(self.doc, self.jobs, start=self.start)
            elif style == "DICT":
                out = self.assemble_output_xml(self.doc, self.jobs, start=self.start)

        # if not sentencized just iterate doc and extract the results
        elif not self.doc.has_annotation("SENT_START"):
            out = self.iterate(out, self.doc, style)
        return out
