# from tracemalloc import start
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.models.sequence_tagger_model import MultiTagger
from flair.data import Token
from tqdm import tqdm
import base as be
import mspacy as msp


class Flair:
    """Base class for Flair, reads in the basic parameters."""

    def __init__(self, config):

        self.outname = config["advanced_options"]["output_dir"] + config["corpus_name"]
        self.input = config["input"]
        self.lang = config["language"]

        config = config["flair_dict"]
        self.job = config["job"]

        # load in a tagger based on the job I guess


class flair_pipe(Flair):
    """Pipeline class for Flair, build pipeline from config and apply it to data.
    Inherits basic parameters from base class."""

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

    # as flair requires sentencized text here is wrapper to a function in
    # mspacy.py which sentencizes a data-string for English or German text

    def senter_spacy(self, data: str) -> Flair:
        """Function to sentencize English or German text using the mspacy module.

        Args:
                data[str]: Input data string containing the text to be sentencized."""

        self.sents = msp.sentencize_spacy(self.lang, data)
        return self

    def apply_to(self) -> Flair:
        """Apply chosen Tagger to data after turning sentences list of strings into list of
        Sentence objects."""

        # wrap the recheived sentence strings from the spacy senter into Sentence-Objects
        self.sentences = [Sentence(sent[0]) for sent in self.sents]
        # apply tagger to list if Sentences
        self.tagger.predict(self.sentences)

        return self

    def get_out(self, ret=False, start=0) -> list or None:
        """Assemble output post-pipeline.

        Args:
                ret[bool]=False: Return output as list (True) or write to file (False).
                start[int]=0: Start index for data. (Maybe not needed?)."""

        out = out_object_flair(self.sentences, self.job, start=0).start_output().out
        for sent in self.sentences:

            out.append("<s>\n")
            out = out_object_flair(sent, self.job, start=0).iterate_tokens(out)
            out.append("</s>\n")

        if ret:
            return out

        elif not ret:
            be.out_object.write_vrt(self.outname, out)
            # be.encode_corpus.encode_vrt("test", self.outname, self.job, "flair")

    def get_multiple(self, chunks: list, ret=False) -> list or None:
        """Iterate through a list of chunks generated by be.chunk_sample_text, tag the tokens
        and create output for full corpus, either return as list or write to .vrt.

        Args:
                chunks[list[list[str]]]: List of chunks which are lists containing
                                                    [opening <>, text, closing <>].

                ret[bool]=False: Wheter to return output as list (True) or write to file (False)."""

        self.sentences = None
        out = out_object_flair(self.sentences, self.job, start=0).start_output().out
        # iterate through the chunks
        for chunk in tqdm(
            chunks,
            total=len(chunks),
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            unit="chunks",
            position=0,
            leave=True,
        ):
            # get the "< >" opening statement
            if chunk[0] != "":
                out.append(chunk[0] + "\n")

            tmp_ = self.senter_spacy(chunk[1]).apply_to()
            for sent in tmp_.sentences:

                out = out_object_flair(sent, tmp_.job, start=0).iterate_tokens(out)

            # append the "< >" closing statement
            if chunk[2] != "":
                out.append(chunk[2] + "\n")

        # either return or write to file
        if ret:
            return out

        elif not ret:
            flat_out = []
            for chunk in out:
                flat_out.append(chunk)
            be.out_object.write_vrt(self.outname, out)
            # be.encode_corpus.encode_vrt("test_chunks", self.outname, self.job, "flair")

    # I guess we will need these more or less for every module separately as the
    # internal structure of the returned objects varies...

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


class out_object_flair(be.out_object):
    """Postprocessing class for Flair. Inherits base out_object."""

    def __init__(self, doc, jobs, start) -> None:
        super().__init__(doc, jobs, start)
        self.job = jobs
        self.sentences = doc
        self.out = []

    def start_output(self: Flair):
        """Initialize the output list."""

        self.out.append("!")

        if type(self.job) == str:
            self.out[0] += " " + self.job

        elif type(self.job) == list:
            for job in self.job:
                self.out[0] += " " + job

        self.out[0] += "\n"

        return self

    def iterate_tokens(self, out: list) -> list:
        """Iterate through a sentence."""

        for token in self.sentences:
            self.assemble_output(token, out)

        return out

    def assemble_output(self, token: Token, out: list) -> list:
        """Build output line from a token.

        Args:
                token[Token]: Annotated token.
                out[list]: Assembled output."""

        out.append("{}".format(token.text))
        for job in self.job:
            label = token.get_label(job).value
            if label != "O":
                out[-1] += " " + label
            elif label == "O":
                out[-1] += " - "

        out[-1] += "\n"
        return out
