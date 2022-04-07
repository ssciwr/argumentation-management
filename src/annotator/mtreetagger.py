import treetaggerwrapper as ttw
import annotator.base as be


def tokenize(text: str, lang: str) -> str and bool:
    """Function to tokenize text using treetaggerwrapper.TreeTagger.tag_text. Text is provided as string.

    [Args]:
            text[str]: String of text to be tokenized.
            lang[str]: Two-char language code, i.e. "en" for english or "de" for german."""

    # load the tokenizer
    tokenizer = ttw.TreeTagger(TAGLANG=lang)
    # tokenize the text
    tokenized = [
        token
        for token in tokenizer.tag_text(
            text,
            prepronly=True,
            notagurl=True,
            notagemail=True,
            notagip=True,
            notagdns=True,
        )
        if token
    ]

    # convert to string in vrt format
    out = ""

    for token in tokenized:
        out += token + "\n"

    # replace problematic patterns
    out = be.out_object.purge(out)
    # text is not sentencized
    sentencized = False

    return out, sentencized


class treetagger_pipe:
    """Class to enable the usage of the treetagger for POS and Lemma tagging
    and subsequent encoding of the results into CWB."""

    def __init__(self, config: dict):
        self.lang = config["lang"]

        tagopt = config["tagopt"]

        self.tagger = ttw.TreeTagger(TAGLANG=self.lang, TAGOPT=tagopt)

    def apply_to(self, text: str) -> list:

        tags = self.tagger.tag_text(
            text, notagurl=True, notagemail=True, notagip=True, notagdns=True
        )
        self.doc = [item._asdict() for item in ttw.make_tags(tags)]

        return self

    def pass_results(self, mydict: dict, style: str) -> None:

        self.jobs = [key for key in self.doc[0].keys()]

        if style == "STR":

            obj = out_object_treetagger(self.doc, self.jobs)

            out = []
            out = obj.iterate(out)

            ptags = obj.get_ptags()
            stags = obj.get_stags()

            outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]
            out_object_treetagger.write_vrt(outfile, out)
            be.encode_corpus.encode_vrt(mydict, ptags, stags)

        elif style == "DICT":
            be.out_object.write_xml(
                mydict["advanced_options"]["output_dir"],
                mydict["corpus_name"],
                self.doc,
            )


class out_object_treetagger(be.out_object):
    def __init__(self, doc, jobs: list, start: int = 0) -> None:
        super().__init__(doc, jobs, start)
        self.doc = doc
        self.attrnames = self.attrnames["treetagger_names"]

    def iterate(self, out: list) -> list or dict:

        for mydict in self.doc:
            out.append("")
            for i, (_, item) in enumerate(mydict.items()):
                if i == 0:
                    out[-1] += item
                elif i > 0:
                    out[-1] += "\t" + item
            out[-1] += "\n"
        return out
