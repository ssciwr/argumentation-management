from flair.data import Sentence
from flair.models import SequenceTagger, MultiTagger
import nlpannotator.base as be


class MyFlair:
    """Flair main processing class. Flair only does POS and NER tagging.

    Args:
       subdict (dictionary): The treetagger input dictionary.
       text (string): The raw text that is to be processed, sentence level or below.
       annotated (object): The output object with annotated tokens.
    """

    def __init__(self, subdict: dict):
        # flair dict
        self.subdict = subdict
        self.jobs = self.subdict["processors"]
        self.model = self.subdict["model"]
        # Initialize the pipeline - only one type of annotation
        if type(self.jobs) == str:
            self.nlp = SequenceTagger.load(self.model)
        elif type(self.jobs) == list:
            self.nlp = MultiTagger.load(self.model)

    def apply_to(self, text: str) -> object:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        # Flair needs the input as sentence object
        self.doc = Sentence(text)
        self.nlp.predict(self.doc)
        return self


class OutFlair(be.OutObject):
    """Out object for flair annotation, adds flair-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int = 0, style: str = "STR") -> None:
        super().__init__(doc, jobs, start, style)
        self.attrnames = self.attrnames["flair_names"]
        self.stags = self.get_stags()
        self.out = []

    def assemble_output_tokens(self, out) -> list:
        """Combine tokens and annotations."""
        # check for list of docs -> list of sentences
        # had been passed that were annotated
        # each sentence (entry in the list) is a flair sentence object
        token_list = []
        if type(self.doc) == list:
            # multiple sentences
            token_list = self.sentence_token_list(self.doc)
        else:
            # only one sentence
            print(type(self.doc))
            token_list += self.token_list(self.doc)

        out = self.iterate_tokens(out, token_list)
        return out

    def grab_tag(self, word):
        """Get pos for token."""
        # attributes:
        # Tagger -> Token.tag, Token.tag_
        tag = word.get_label(self.attrnames["pos"]).value
        return tag

    def sentence_token_list(self, myobj):
        """Flair-specific iteration through sentences and tokens simultaneously."""
        st_list = []
        for sentence in myobj:
            for token in sentence:
                st_list.append(token)
        return st_list
