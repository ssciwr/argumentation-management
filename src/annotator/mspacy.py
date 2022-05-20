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
        self.jobs = subdict["processors"]
        # we need to map the jobs to spacy notation
        self.model = subdict["model"]

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

        # use specific device settings if requested
        # this also to be set in the pipeline decision
        if subdict["set_device"]:
            if subdict["set_device"] == "prefer_GPU":
                sp.prefer_gpu()
            elif subdict["set_device"] == "require_GPU":
                sp.require_gpu()
            elif subdict["set_device"] == "require_CPU":
                sp.require_cpu()

        self.config = subdict["config"]


# build the pipeline from config-dict
class spacy_pipe(MySpacy):
    """Assemble pipeline from config, apply pipeline to data and write results to .vrt file."""

    # init with specified config, this may be changed later?
    # -> Right now needs quite specific instuctions
    def __init__(self, config: dict):
        super().__init__(config)
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

    # call the build pipeline on the data
    def apply_to(self, data: str) -> Doc:
        """Apply the objects pipeline to a given data string."""

        # apply to data while disabling everything that wasnt requested
        self.doc = self.nlp(data)
        return self

    def pass_results(
        self,
        mydict: dict or None = None,
        style: str = "STR",
        ret: bool = False,
        start: int = 0,
        add: bool = False,
        ptags: list or None = None,
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
        ptags = ptags or out_obj.ptags
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


if __name__ == "__main__":
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    mydict["processing_option"] = "accurate"
    mydict["processing_type"] = "sentencize, pos  ,lemma, tokenize"
    obj = pe.SetConfig(mydict)
    be.prepare_run.validate_input_dict(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
    spacy_dict = obj.mydict["spacy_dict"]
    # load the pipeline from the config
    pipe = spacy_pipe(spacy_dict)
    data = be.prepare_run.get_text("./src/annotator/test/test_files/example_de.txt")
    # apply pipeline to data
    annotated = pipe.apply_to(data)
    # get the dict for encoding
    # encoding_dict = be.prepare_run.get_encoding(mydict)
    # Write vrt and encode
    annotated.pass_results(mydict)
