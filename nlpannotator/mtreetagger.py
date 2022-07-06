import treetaggerwrapper as ttw
import nlpannotator.base as be


class MyTreetagger:
    """Treetagger main processing class.

    Args:
       subdict (dictionary): The treetagger input dictionary.
       text (string): The raw text that is to be processed, sentence level or below.
       annotated (object): The output object with annotated tokens.
    """

    def __init__(self, subdict: dict):
        # treetagger dict
        self.subdict = subdict
        self.jobs = self.subdict["processors"]
        # set tagonly if already tokenized
        if "tokenize" not in self.jobs:
            self.subdict["tagonly"] = True
        # Initialize the pipeline
        self.nlp = ttw.TreeTagger(
            TAGLANG=self.subdict["lang"], TAGOPT=self.subdict["tagopt"]
        )

    def apply_to(self, text: str) -> object:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp.tag_text(text)
        # separate the tags and put in dict - use treetagger intrinsic
        self.doc = self._make_dict()
        # convert the dict to properties of token object
        self._make_object()
        return self

    def _make_dict(self):
        # separate the tags and put in dict
        return [item._asdict() for item in ttw.make_tags(self.doc)]

    def _make_object(self):
        self.doc = [TreetaggerDoc(token) for token in self.doc]
        return self


class TreetaggerDoc:
    def __init__(self, outdict: dict):
        self.outdict = outdict

    @property
    def text(self):
        return self.outdict["word"]

    @property
    def lemma(self):
        return self.outdict["lemma"]

    @property
    def pos(self):
        return self.outdict["pos"]


class OutTreetagger(be.OutObject):
    """Out object for treetagger annotation, adds treetagger-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int = 0, style: str = "STR") -> None:
        super().__init__(doc, jobs, start, style)
        self.attrnames = self.attrnames["treetagger_names"]
        self.stags = self.get_stags()

    def assemble_output_tokens(self, out) -> list:
        # check for list of docs -> list of sentences
        # had been passed that were annotated
        token_list = []
        if type(self.doc) == list:
            token_list += self.token_list(self.doc)
        else:
            print(type(self.doc))

        out = self.iterate_tokens(out, token_list)
        return out
