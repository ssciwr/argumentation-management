import spacy as sp
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from spacy.lang.de import German
import copy
import base as be
import pipe as pe
from tqdm import (
    tqdm,
)  # for progress in pipe_multiple, might be interesting for large corpora down the line


class MySpacy:
    """Base class for spaCy module.

    Args:
        subdict[dict]: Dict containing the setup for the spaCy run.
    """

    def __init__(self, subdict: dict):
        # set the jobs for spacy
        self.jobs = subdict["processors"]
        # set the model for spacy
        self.model = subdict["model"]
        # activate tok2vec to boost accuracy
        self._set_tok2vec()
        # activate GPU usage if preferred
        self._set_device(subdict)
        # set spacy parameters from spacy_dict["config"] dictionary
        self.config = subdict["config"]
        # load the pipeline
        self._load_pipe()

    def _load_pipe(self):
        # use a specific pipeline if requested
        self.validated = []
        try:
            self.nlp = sp.load(self.model, config=self.config)

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

    def _set_tok2vec(self):
        # if we ask for lemma and/or POS we force tok2vec to boost accuracy
        # also add in attribute ruler as it is cheap
        if (
            "lemmatizer" in self.jobs
            or "tagger" in self.jobs
            or "lemmatizer"
            and "tagger" in self.jobs
        ):
            if "tok2vec" not in self.jobs:
                self.jobs = ["tok2vec"] + self.jobs
            if "attribute_ruler" not in self.jobs:
                self.jobs.append("attribute_ruler")

    def _set_device(self, subdict):
        # use specific device settings if requested
        # this also to be set in the pipeline decision
        if subdict["set_device"]:
            if subdict["set_device"] == "prefer_GPU":
                sp.prefer_gpu()
            elif subdict["set_device"] == "require_GPU":
                sp.require_gpu()
            elif subdict["set_device"] == "require_CPU":
                sp.require_cpu()

    # call the build pipeline on the data
    def apply_to(self, data: str) -> Doc:
        """Apply the objects pipeline to a given data string."""

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data)
        return self

    # sentencizer only
    @staticmethod
    def sentencize_spacy(model: str, data: str) -> list:
        """Function to sentencize given text data.

        Args:
                data[str]: The text string to be split into sentences.

        Returns:
                List[List[str, int]]: List containing lists which contain the sentences as strings
                as well as the number of tokens previous to the sentence to easily keep
                track of the correct token index for a given sentence in the list.
        """
        nlp = sp.load(model)
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
class OutSpacy(be.OutObject):
    """Out object for spacy annotation, adds spacy-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs, start, islist=False):
        super().__init__(doc, jobs, start, islist)
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

    # this to be removed as it duplicates functionality - TODO
    def fetch_output(self, style) -> list:
        """Function to assemble the output list for a run. Can work with or without sentence
        level annotation and will check if doc is sentencized on its own."""

        try:
            assert hasattr(self, "doc")
        except AttributeError:
            print(
                "Seems there is no Doc object, did you forget to call MySpacy.apply_to()?"
            )
            exit()

        out = []
        # check if spacy doc object is sentencized
        if self.doc.has_annotation("SENT_START"):
            # apply sentence and sublevel annotation
            if style == "STR":
                out = self.assemble_output_sent()
            elif style == "DICT":
                out = self.assemble_output_xml()

        # if not sentencized just iterate doc and extract the results
        elif not self.doc.has_annotation("SENT_START"):
            out = self.iterate(out, self.doc, style)
        return out

    @property
    def sentences(self) -> list:
        """Function to return sentences as list.

        Returns:
                List[List[str, int]]: List containing lists which contain the sentences as strings
                as well as the number of tokens previous to the sentence to easily keep
                track of the correct token index for a given sentence in the list.
        """

        try:
            assert hasattr(self, "doc")
        except AttributeError:
            print(
                "Seems there is no Doc object, did you forget to call MySpacy.apply_to()?"
            )
            exit()
        assert self.doc.has_annotation("SENT_START")

        sents = []
        for i, sent in enumerate(self.doc.sents):
            if i == 0:
                # need to take the len of the split str as otherwise grouping of multiple tokens by
                # spacy can be a problem. This now assumes that tokens are always separated by a
                # whitespace, which seems reasonable to me -> Any examples to the contrary?
                sents.append([sent.text, len(sent.text.split())])
            elif i > 0:
                sents.append([sent.text, len(sent.text.split()) + sents[i - 1][1]])
        return sents
