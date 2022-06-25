# from tracemalloc import start
from flair.data import Sentence
from flair.models import SequenceTagger, MultiTagger
from flair.data import Token
import base as be


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
        if len(self.jobs) == 1:
            self.nlp = SequenceTagger.load(self.model)
        elif len(self.jobs) > 1:
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

    def __init__(self, doc, jobs: list, start: int = 0, islist=False) -> None:
        super().__init__(doc, jobs, start, islist)
        self.attrnames = self.attrnames["flair_names"]
        self.ptags = self.get_ptags()
        self.stags = None
        self.out = []

    def assemble_output_tokens(self, out) -> list:
        # check for list of docs -> list of sentences
        # had been passed that were annotated
        # each sentence (entry in the list) is a flair sentence object
        token_list = []
        if type(self.doc) == list:
            token_list += self.token_list(self.doc)
        else:
            print(type(self.doc))
            token_list += self.token_list(self.doc)

        token_list_out = self.out_shortlist(out)
        # now compare the tokens in out with the token objects from treetagger
        # here treetagger is a bit different since it is not a list of objects
        # but a list of dict
        for token_treetagger, token_out in zip(token_list, token_list_out):
            mylen = len(token_treetagger.text)
            print(
                "Checking for tokens {} {}".format(token_treetagger.text, token_out[0])
            )
            # check that the text is the same
            if token_treetagger.text != token_out[0][0:mylen]:
                print(
                    "Found different token than in out! - {} and {}".format(
                        token_treetagger.text, token_out[0][0:mylen]
                    )
                )
                print("Please check your inputs!")
            else:
                line = self.collect_results(
                    token_treetagger, 0, token_treetagger, "STR"
                )
                # now replace the respective token with annotated token
                out[token_out[1]] = out[token_out[1]].replace("\n", "") + line + "\n"
                # note that this will not add a linebreak for <s> and <s\> -
                # linebreaks are handled by sentencizer
                # we expect that sentencizer runs TODO be able to feed only one sentence
        return out

    def grab_tag(self, word):

        # attributes:
        # Tagger -> Token.tag, Token.tag_
        if word.get_label(self.attrnames["pos"]).value != "0":
            tag = word.get_label(self.attrnames["pos"]).value
        else:
            tag = "NOT_DEF"
        return tag


if __name__ == "__main__":
    data = "This is a sentence."
    mydict = be.prepare_run.load_input_dict("annotator/input")
    mydict["tool"] = "flair"
    mydict["advanced_options"]["output_dir"] = "annotator/test/out/"
    mydict["advanced_options"]["corpus_dir"] = "annotator/test/corpora/"
    mydict["advanced_options"]["registry_dir"] = "annotator/test/registry/"
    flair_dict = mydict["flair_dict"]
    flair_dict["processors"] = "tokenize", "pos"
    flair_dict["model"] = ["pos", "ner"]
    annotated = MyFlair(flair_dict)
    annotated = annotated.apply_to(data)
    start = 0
    out_obj = OutFlair(annotated.doc, annotated.jobs, start=start, islist=False)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    print(out)
    ptags = out_obj.get_ptags()
    stags = out_obj.get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    # add = False
    # if not add:
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
