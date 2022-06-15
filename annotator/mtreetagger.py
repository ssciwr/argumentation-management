import treetaggerwrapper as ttw
import base as be
import pprint


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
        # Initialize the pipeline
        self.nlp = ttw.TreeTagger(
            TAGLANG=self.subdict["lang"], TAGOPT=self.subdict["tagopt"]
        )

        # self.nlp = sa.Pipeline(**self.subdict)

    def apply_to(self, text: str) -> dict:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp.tag_text(text)
        self.doc = [item._asdict() for item in ttw.make_tags(self.doc)]

        return self

    # tokenized = [
    # token
    # for token in tokenizer.tag_text(
    # text,
    # prepronly=True,
    # notagurl=True,
    # notagemail=True,
    # notagip=True,
    # notagdns=True,
    # )
    # if token
    # ]

    # for token in tokenized:
    # out += token + "\n"

    # if style == "STR":
    # obj = out_object_treetagger(self.doc, self.jobs)


#
# out = []
# grab the tagged string
# out = obj.iterate(out, None, "STR")
#
# check for tags for encoding, for this tool it should be POS and Lemma
# ptags = obj.get_ptags()
# stags = obj.get_stags()
#
# outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
#
# if not add:
# out_object_treetagger.write_vrt(outfile, out)
# encode_obj = be.encode_corpus(mydict)
# encode_obj.encode_vrt(ptags, stags)
#
# elif add:
# be.OutObject.write_vrt(outfile, out)
# encode_obj = be.encode_corpus(mydict)
# encode_obj.encode_vrt(ptags, stags)
#
# elif style == "DICT":
# be.OutObject.write_xml(
# mydict["advanced_options"]["output_dir"],
# mydict["corpus_name"],
# self.doc,
# )


class out_object_treetagger(be.OutObject):
    """Class to define how information will be extracted from the doc object
    resulting from the treetagger pipeline."""

    def __init__(self, doc, jobs: list, start: int = 0) -> None:
        super().__init__(doc, jobs, start)
        self.doc = doc
        self.attrnames = self.attrnames["treetagger_names"]

    def iterate(self, out: list, sent, style) -> list:
        """Iterate the list of tagged tokens and extract the information for
        further processing."""

        for mydict in self.doc:
            out.append("")
            for i, (_, item) in enumerate(mydict.items()):
                if i == 0:
                    out[-1] += item
                elif i > 0:
                    out[-1] += "\t" + item
            out[-1] += "\n"
        return out


if __name__ == "__main__":
    data = "This is a sentence."
    mydict = be.prepare_run.load_input_dict("annotator/input")
    mydict["tool"] = "treetagger"
    treetagger_dict = mydict["treetagger_dict"]
    treetagger_dict["lang"] = "en"
    treetagger_dict["processors"] = "tokenize", "pos", "lemma"
    annotated = MyTreetagger(treetagger_dict)
    doc = annotated.apply_to(data)
    print(doc)
