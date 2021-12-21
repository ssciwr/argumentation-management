from logging import StringTemplateStyle
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
        assert doc.has_annotation("SENT_START")

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
            # print(self.sents[-1])
            # if self.sents[i-1][1] > 500:
            #    break
        return self

    def apply_to(self):
        """Apply chosen Tagger to data after turning sentences list of strings into list of
        Sentence objects."""

        self.sentences = [Sentence(sent[0]) for sent in self.sents]
        self.tagger.predict(self.sentences)

        return self

    def get_multiple(self, chunks: list, ret=False):
        """Iterate through a list of chunks generated by be.chunk_sample_text, tag the tokens
        and create output for full corpus, either return as list or write to .vrt.

        [Args]:
                chunks[list[list[str,str,str]]]: List of chunks which are lists containing
                    [opening <>, text, closing <>].
                ret[bool]=False: Wheter to return output as list (True) or write to file (False)."""

        # Initialize the out list
        out = self.start_output()

        # iterate through the chunks
        for i, chunk in enumerate(chunks):
            # get the "< >" opening statement
            out.append(data1[i][0] + "\n")
            if i == 0:
                # sentencize chunk and apply tagger, token index from 0
                tmp = self.sentencize_spacy(chunk[1]).apply_to().to_vrt(ret=True)
            elif i > 0:
                # sentencize chunk, apply tagger, keeping token index from previous chunk
                tmp = (
                    self.sentencize_spacy(chunk[1])
                    .apply_to()
                    .to_vrt(ret=True, start=be.find_last_idx(tmp) + 1)
                )

            tokens = 0
            # append data from tmp pipe output to complete output,
            # maybe count the tokens for failchecking later? -> Token
            # idx should coincide with the cwb corpus index for final
            # output?
            for line in tmp:
                if not line.startswith("!") and not line.startswith("<"):
                    tokens += 1
                out.append(line)
            # append the "< >" closing statement
            out.append(data1[i][2] + "\n")
            print(
                "\r"
                + " Finished chunk {}/{}, {} Token".format(i + 1, len(data1), tokens)
            )

        # either return or write to file
        if ret:
            return out

        elif not ret:
            with open("{}_flair.vrt".format(self.filename), "w") as file:
                for chunk in out:
                    for line in chunk:
                        file.write(line)

    def start_output(self) -> list:
        """Initialize the output list."""

        out = ["! Output flair", "! Idx Token"]

        if type(self.job) == str:
            out[1] += " " + self.job

        elif type(self.job) == list:
            for job in self.job:
                out[1] += " " + job

        out[0] += "\n"
        out[1] += "\n"

        return out

    def assemble_output(self, token, out, current):
        out.append("{} {}".format(current, token.text))
        for label in token.labels:
            if label.value != "O":
                out[-1] += " " + label.value
            elif label.value == "O":
                out[-1] += " - "

        return out

    # I guess we will need these more or less for every module separately as the
    # internal structure of the returned objects varies...
    def to_vrt(self, ret=False, start=0):

        out = self.start_output()
        screwed = False
        for i, sent in enumerate(self.sentences):
            out.append("<s>\n")

            # check if any indices got scrambled because flair tokenized differently
            # than spacy

            if i > 0 and i < len(self.sents) - 1 and not screwed:
                try:
                    if self.sents[i + 1][1] - self.sents[i - 1][1] != len(sent):
                        self.sents[i][1] = self.sents[i - 1][1] + len(sent)
                        screwed = True
                except IndexError:
                    print("Error at ", i)
                    print(self.sents[i - 1][1])

            # if the index got scrambled somewhere we have to rebuild it on the fly and
            # cant just look it up
            elif screwed:
                self.sents[i][1] = self.sents[i - 1][1] + len(sent)

            for j, token in enumerate(sent):

                if i == 0:
                    out = self.assemble_output(token, out, j + start)
                    # print(j, start)
                elif i > 0:
                    out = self.assemble_output(
                        token, out, j + start + self.sents[i - 1][1]
                    )
                    # print( j, start, self.sents[i-1][1])
                out[-1] += "\n"

            out.append("</s>\n")

        if ret:
            return out

        elif not ret:
            with open("{}_flair.vrt".format(self.filename), "w") as file:
                for line in out:
                    file.write(line)

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
            if not line.startswith("!") and not line.startswith("<"):
                check.append(int(line.split()[0]))

    for i, elem in enumerate(check):
        if i > 0:
            try:
                assert elem - check[i - 1] == 1
            except AssertionError:
                print(i, elem)

    data1 = be.chunk_sample_text("data/Original/plenary.vrt")

    cfgman = {"filename": "testplenary", "lang": "de", "job": ["pos", "ner"]}

    flair_pipe(cfgman).get_multiple(data1)

    # index check -> gibt noch ein paar Probleme, bin dran
    # // scheint gefixed zu sein, schaue am Donnerstag nochmal drauf
    check = []
    with open("testplenary_flair.vrt", "r") as file:
        for line in file:
            if not line.startswith("!") and not line.startswith("<"):
                check.append(int(line.split()[0]))

    for i, elem in enumerate(check):
        if i > 0:
            try:
                assert elem - check[i - 1] == 1
            except AssertionError:
                print(i, elem)
