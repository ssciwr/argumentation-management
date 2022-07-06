from somajo import SoMaJo
import nlpannotator.base as be


class MySomajo:
    """Class to tokenize text using somajo.tokenize_text. Text may be provided as
    string or as list of strings.

    [Args]:
           text[list[str] or str]: List of strings (paragraphs) or string.
           model[str]: Model to be used by somajo, options are de_CMC, en_PTB.
           split_sentences[bool]: Perform sentence splitting in addition to tokenization."""

    def __init__(self, subdict: dict) -> None:
        self.model = subdict["model"]
        self.jobs = subdict["processors"]
        self.sentencize = subdict["split_sentences"]
        self.camelcase = subdict["split_camel_case"]

    def apply_to(self, text: list or str):
        """Apply pipeline to text."""
        # somajo takes list as input
        if type(text) == str:
            text = [text]

        self.doc = list(
            list(
                SoMaJo(
                    self.model,
                    split_camel_case=self.camelcase,
                    split_sentences=self.sentencize,
                ).tokenize_text(text)
            )
        )
        return self


class OutSomajo(be.OutObject):
    def __init__(self, doc, jobs, start, style: str = "STR"):
        super().__init__(doc, jobs, start, style)
        self.attrnames = self.attrnames["somajo_names"]
        self.stags = self.get_stags()

    def assemble_output_sent(self) -> list:
        """Sentence assembly for somajo."""

        # if senter is called we insert sentence symbol <s> before and </s> after
        # every sentence
        # if only sentence is provided, directly call the methods

        self.tstart = 0
        out = []
        for sent in self.doc:
            out.append("<s>\n")
            out = self.iterate(out, sent)
            out.append("</s>\n")
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

        sents = []
        for sent in self.doc:
            line = ""
            for token in sent:
                line += token.text + " "
            line = line.strip()
            sents.append(line)
        return sents
