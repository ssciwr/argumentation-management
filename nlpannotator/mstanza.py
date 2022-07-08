# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import nlpannotator.base as be


class MyStanza:
    """Stanza main processing class.

    Args:
       subdict (dictionary): The stanza input dictionary.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (object): The output object with annotated tokens.
    """

    def __init__(self, subdict: dict):
        # stanza dict
        self.subdict = subdict
        if "," in self.subdict["processors"]:
            self.jobs = self.subdict["processors"].split(",")
        else:
            self.jobs = self.subdict["processors"]
        # Initialize the pipeline
        self.nlp = sa.Pipeline(**self.subdict)

    def apply_to(self, text: str) -> dict:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp(text)  # Run the pipeline on the input text
        return self


# to be integrated in collect results - TODO
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
class OutStanza(be.OutObject):
    """Out object for stanza annotation, adds stanza-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int = 0, style: str = "STR"):
        super().__init__(doc, jobs, start, style)
        self.attrnames = self.attrnames["stanza_names"]
        self.stags = self.get_stags()

    # add new method for stanza iteration over tokens/words/ents
    # TODO: set MWT correctly - iterate over tokens or words?
    def iterate(self, out: list, sent) -> list:
        """Function to iterate through sentence object and extract data to list.

        Args:
                out[list]: List containing the collected output.
                sent[stanza sent-Object]: Object containing tokenized sentence."""

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
            line = token.text
            out.append(line + "\n")
        self.tstart = tid
        return out

    def assemble_output_tokens(self, out) -> list:
        # for stanza we always have sentence level
        # as stanza allows feeding of sentences manually
        token_list = []
        word_list = []
        for sent in self.doc.sentences:
            token_list += self.token_list(sent)
            word_list += self.word_list(sent)
        token_list_out = self.out_shortlist(out)
        # now compare the tokens in out with the token objects from stanza
        # here we need to check what to do with mwt - we may need word
        # instead of token
        for token_stanza, word_stanza, token_out in zip(
            token_list, word_list, token_list_out
        ):
            mylen = len(token_stanza.text)
            # print(
            # "Checking for tokens {} {}".format(
            # token_stanza.text, token_out[0][0:mylen]
            # )
            # )
            # print(
            # "Checking for words {} {}".format(
            # word_stanza.text, token_out[0][0:mylen]
            # )
            # )
            # check that the text is the same
            # here we may need to check for word..?
            if token_stanza.text != token_out[0][0:mylen]:
                print(
                    "Found different token than in out! - {} and {}".format(
                        token_stanza.text, token_out[0][0:mylen]
                    )
                )
                print("Please check your inputs!")
            else:
                line = self.collect_results(token_stanza, 0, word_stanza)
                # now add the annotation
                out[token_out[1]] = out[token_out[1]].replace("\n", "") + line + "\n"
        return out

    def token_list(self, myobj: list) -> list:
        return [token for token in myobj.tokens]

    def word_list(self, myobj: list) -> list:
        return [word for word in myobj.words]

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
        for sent in self.doc.sentences:
            sents.append(sent.text)
        return sents
