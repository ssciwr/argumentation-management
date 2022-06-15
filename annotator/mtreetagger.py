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


class OutTreetagger(be.OutObject):
    """Out object for treetagger annotation, adds treetagger-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int = 0, islist=False) -> None:
        super().__init__(doc, jobs, start, islist)
        self.attrnames = self.attrnames["treetagger_names"]
        self.ptags = self.get_ptags()
        self.stags = None

    # add new method for treetagger iteration over tokens
    def iterate(self, out: list, sent, style) -> list:
        """Function to iterate through sentence object and extract data to list.

        Args:
                out[list]: List containing the collected output.
                sent[treetagger sent-Object]: Object containing tokenized sentence."""

        for mydict in self.doc:
            out.append("")
            for i, (_, item) in enumerate(mydict.items()):
                if i == 0:
                    out[-1] += item
                elif i > 0:
                    out[-1] += "\t" + item
            out[-1] += "\n"
        return out

    def assemble_output_tokens(self, out) -> list:
        # check for list of docs -> list of sentences
        # had been passed that were annotated
        token_list = []
        # if we feed sentences, senter and parser processors need to be absent
        # apparently nothing else
        # see https://spacy.io/api/doc#sents
        if type(self.doc) == list:
            token_list += self.token_list(self.doc)
        else:
            print(type(self.doc))

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
        return out

    def token_list(self, myobj) -> list:
        return [token for token in myobj]

    def out_shortlist(self, out: list) -> list:
        out = [
            (token.strip(), i)
            for i, token in enumerate(out)
            if token.strip() != "<s>" and token.strip() != "</s>"
        ]
        return out

    def _compare_tokens(self, token1, token2):
        return token1 == token2


if __name__ == "__main__":
    data = "This is a sentence."
    mydict = be.prepare_run.load_input_dict("annotator/input")
    mydict["tool"] = "treetagger"
    treetagger_dict = mydict["treetagger_dict"]
    treetagger_dict["lang"] = "en"
    treetagger_dict["processors"] = "tokenize", "pos", "lemma"
    annotated = MyTreetagger(treetagger_dict)
    annotated = annotated.apply_to(data)
    start = 0
    out_obj = OutTreetagger(annotated.doc, annotated.jobs, start=start, islist=False)
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
