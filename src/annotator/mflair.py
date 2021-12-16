from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models.sequence_tagger_model import MultiTagger
from spacy.lang.en import English
from spacy.lang.de import German
import base as be


class flair:
    def __init__(self, config):

        self.filename = config["filename"]
        self.lang = config["lang"]
        self.job = config["job"]

        # load in a tagger based on the job I guess


class flair_pipe(flair):

    # initialize flair NER by loading the tagger specified by lang
    def __init__(self, lang):
        super().__init__(lang)

        # first, check for language
        if self.lang == "de":

            # then check for type of input, is it a str or a list of strings
            if type(self.job) == str:

                # if it is a string we will only need the normal sequence tagger
                if self.job == "ner":
                    self.tagger = SequenceTagger.load("de-ner")
                elif self.job == "pos":
                    self.tagger = SequenceTagger.load("de-pos")

            # if it is not a str it should be a list of strings
            elif type(self.job) == list:

                # we will need the MultiTagger -> can not just use the passed arguments here
                # as the keywords for german standard models are different
                # -> maybe allow specification of certain models by the user later on?
                self.tagger = MultiTagger.load(["de-pos", "de-ner"])

        # same stuff for english
        elif self.lang == "en":

            if type(self.job) == str:
                if self.job == "ner":
                    self.tagger = SequenceTagger.load("ner")
                elif self.job == "pos":
                    self.tagger = SequenceTagger.load("pos")

            elif type(self.job) == list:
                self.tagger = MultiTagger.load(self.job)

    # as flair requires sentencized text here is a small function which
    # can sentencize English and German text using spaCy. This is a class
    # function here but we could export it somewhere else as well and just
    # pass the created list to the flair object rather than have it build
    # into it
    def sentencize_spacy(self, data: str):
        """Function to sentencize given text data in either German or English language.

        [args]:
                data[str]: The text string to be split into sentences.

        [returns]:
                List[List[str, int]]: List containing lists which contain the sentences as strings
                    as well as the number of tokens previous to the sentence to easily keep
                    track of the correct token index for a given sentence in the list."""

        if self.lang == "en":
            nlp = English()
            nlp.add_pipe("sentencizer")

        elif self.lang == "de":
            nlp = German()
            nlp.add_pipe("sentencizer")

        doc = nlp(data)
        self.sents = []

        for i, sent in enumerate(doc.sents):
            if i == 0:
                # need to take the len of the split str as otherwise grouping of multiple tokens by
                # spacy can be a problem. This now assumes that tokens are always separated by a
                # whitespace, which seems reasonable to me -> Any examples to the contrary?
                self.sents.append([sent.text, len(sent.text.split())])
            elif i > 0:
                self.sents.append(
                    [sent.text, len(sent.text.split()) + self.sents[i - 1][1]]
                )

        return self

    def apply_to(self):

        self.sentences = [Sentence(sent[0]) for sent in self.sents]
        self.tagger.predict(self.sentences)

        return self

    # I guess we will need these more or less for every module separately as the
    # internal structure of the returned objects varies...
    def to_vrt(self):

        out = ["! Output flair", "! Idx Token"]

        if type(self.job) == str:
            out[0] += " " + self.job
            out[1] += " " + self.job

        elif type(self.job) == list:
            for job in self.job:
                out[0] += " " + job
                out[1] += " " + job

        for i, sent in enumerate(self.sentences):
            for j, token in enumerate(sent):

                if i == 0:
                    out.append("{} {}".format(j, token.text))
                    for label in token.labels:
                        if label.value != "O":
                            out[-1] += " " + label.value
                        elif label.value == "O":
                            out[-1] += " - "

                elif i > 0:
                    out.append("{} {}".format(j + self.sents[i - 1][1], token.text))
                    for label in token.labels:
                        if label.value != "O":
                            out[-1] += " " + label.value
                        elif label.value == "O":
                            out[-1] += " - "

        with open("{}_flair.vrt".format(self.filename), "w") as file:
            for line in out:
                file.write(line + "\n")

    # def __call__(self, tokens):
    #    sentences = [Sentence(tokens)]
    #    self.tagger.predict(sentences)

    #    self.named_entities = defaultdict(list)

    #    for sentence in sentences:
    #        for entity in sentence.get_spans():
    #            self.named_entities[
    #                "Start {} End {}".format(
    #                    entity.start_pos, str(entity.end_pos).split()[0]
    #                )
    #            ].append([entity.text, entity.labels[0]])

    #    return self.named_entities


if __name__ == "__main__":

    cfg = be.load_input_dict()
    flair_cfg = be.update_dict(cfg["flair_dict"])
    data = be.get_sample_text()
    flair_pipe(flair_cfg).sentencize_spacy(data).apply_to().to_vrt()

    # check that the indexing is correct -> no skips or setbacks

    check = []
    with open("Test_flair.vrt", "r") as file:
        for line in file:
            if line.split()[0] != "!":
                check.append(int(line.split()[0]))

    for i, elem in enumerate(check):
        if i > 0:
            try:
                assert elem - check[i - 1] == 1
            except AssertionError:
                print(i, elem)
