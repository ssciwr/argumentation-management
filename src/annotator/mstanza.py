# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be


class MyStanza:
    """Stanza main processing class.

    Args:
       subdict (dictionary): The stanza input dictionary.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (dictionary): The output dictionary with annotated tokens.
    """

    def __init__(self, subdict: dict):
        # stanza dict
        self.subdict = subdict
        self.jobs = self.subdict["processors"].split(",")
        # Initialize the pipeline
        self.nlp = sa.Pipeline(**self.subdict)

    def apply_to(self, text: str) -> dict:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp(text)  # Run the pipeline on the input text
        return self

    # this should not be needed anymore


def ner(doc) -> dict:
    """Function to extract NER tags from Doc Object."""

    named_entities = defaultdict(list)

    for ent in doc.ents:
        # add the entities, key value is start and end char as Token IDs in stanza seem to be restricted to
        # individual sentences
        named_entities["{} : {}".format(ent.start_char, ent.end_char)].append(
            [ent.text, ent.type]
        )

    return named_entities


# inherit the output class from base and add stanza-specific methods
class out_object_stanza(be.OutObject):
    """Out object for stanza annotation, adds stanza-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int = 0, islist=False):
        super().__init__(doc, jobs, start, islist)
        self.attrnames = self.attrnames["stanza_names"]
        self.ptags = self.get_ptags()
        self.stags = self.get_stags()

    # add new method for stanza iteration over tokens/words/ents
    def iterate(self, out: list, sent, style: str) -> list:
        """Function to iterate through sentence object and extract data to list.

        Args:
                out[list]: List containing the collected output.
                sent[stanza sent-Object]: Object containing annotated sentence."""

        for token, word in zip(getattr(sent, "tokens"), getattr(sent, "words")):
            if token.text != word.text:
                print(
                    "Found MWT - please check if annotated correctly!!! Token {} != word {}.".format(
                        token.text, word.text
                    )
                )
                print("Because I am not sure how CWB handles these s-attributes.")
                # raise NotImplementedError(
                # "Multi-word expressions not available currently"
                # )
            tid = token.id[0] + self.tstart
            line = self.collect_results(token, tid, word, style)

            out.append(line + "\n")
        self.tstart = tid
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
                "Seems there is no Doc object, did you forget to call MyStanza.apply_to()?"
            )
            exit()

        sents = []
        for i, sent in enumerate(self.doc.sentences):
            if i == 0:
                # need to take the len of the split str as otherwise grouping of multiple tokens by
                # spacy can be a problem. This now assumes that tokens are always separated by a
                # whitespace, which seems reasonable to me -> Any examples to the contrary?
                sents.append([sent.text, len(sent.text.split())])
            elif i > 0:
                sents.append([sent.text, len(sent.text.split()) + sents[i - 1][1]])
        return sents
