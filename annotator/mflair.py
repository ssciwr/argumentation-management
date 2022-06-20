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
        print(self.doc)
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
        return


if __name__ == "__main__":
    data = "This is a sentence."
    mydict = be.prepare_run.load_input_dict("annotator/input")
    mydict["tool"] = "flair"
    flair_dict = mydict["flair_dict"]
    flair_dict["processors"] = "tokenize", "pos"
    flair_dict["model"] = ["pos", "ner"]
    annotated = MyFlair(flair_dict)
    annotated = annotated.apply_to(data)
    exit()

    # for sent in self.sentences:
    #
    # out.append("<s>\n")
    # out = out_object_flair(sent, self.job, start=0).iterate_tokens(out)
    # out.append("</s>\n")
    #
    # if ret:
    # return out
    #
    # elif not ret:
    # be.OutObject.write_vrt(self.outname, out)
    # be.encode_corpus.encode_vrt("test", self.outname, self.job, "flair")

    start = 0
    out_obj = OutFlair(annotated.doc, annotated.jobs, start=start, islist=False)
    out = ["<s>", "This", "is", "a", "sentence", ".", "</s>"]
    out = out_obj.assemble_output_tokens(out)
    ptags = out_obj.get_ptags()
    stags = out_obj.get_stags()
    # write out to .vrt
    outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
    out_obj.write_vrt(outfile, out)
    # add = False
    # if not add:
    encode_obj = be.encode_corpus(mydict)
    encode_obj.encode_vrt(ptags, stags)
