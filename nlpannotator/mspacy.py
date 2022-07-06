import spacy as sp
import nlpannotator.base as be


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
    def apply_to(self, data: str):
        """Apply the objects pipeline to a given data string."""

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data)
        return self


# inherit the output class from base and add spacy-specific methods
class OutSpacy(be.OutObject):
    """Out object for spacy annotation, adds spacy-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs, start, style: str = "STR"):
        super().__init__(doc, jobs, start, style)
        self.attrnames = self.attrnames["spacy_names"]
        self.stags = self.get_stags()

    def assemble_output_tokens(self, out) -> list:
        """Assemlbe token and annotation data."""
        # check for list of docs -> list of sentences
        # had been passed that were annotated
        token_list = []
        # if we feed sentences, senter and parser processors need to be absent
        # apparently nothing else
        # see https://spacy.io/api/doc#sents
        if type(self.doc) == list:
            for doc in self.doc:
                token_list += self.token_list(doc)
        # else spacy was used also for sentencizing
        # check if sentence-level is there
        else:
            if self.doc.has_annotation("SENT_START"):
                for sent in self.doc.sents:
                    token_list += self.token_list(sent)
        out = self.iterate_tokens(out, token_list)
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
        for sent in self.doc.sents:
            sents.append(sent.text)
        return sents
