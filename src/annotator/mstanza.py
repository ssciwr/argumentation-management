# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be


class MyStanza:
    """Stanza main processing class.

    Args:
       config (dictionary): The full input dictionary.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (dictionary): The output dictionary with annotated tokens.
    """

    def __init__(self, config: dict):
        # get the stanza dict
        self.config = config
        # Initialize the pipeline using a configuration dict
        self.nlp = sa.Pipeline(**self.config)

    def apply_to(self, text: str) -> dict:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp(text)  # Run the pipeline on the input text
        return self

    def process_multiple_texts(self, textlist: list) -> dict:
        """Function to process multiple texts.

        Args:
                textlist[list]: List containing the texts."""

        # Wrap each document with a stanza.Document object
        in_docs = [sa.Document([], text=d) for d in textlist]
        self.mdocs = self.nlp(
            in_docs
        )  # Call the neural pipeline on this list of documents
        return self.mdocs

    def pass_results(
        self, mydict: dict, add: bool = False, ptags: list or None = None
    ) -> None:
        """Funtion to write post-pipeline data to .vrt file and encode for CWB.

        Args:
                out_param[dict]: Parameters for output.
                add[bool]: Indicates if a new corpus should be started or if tags should be added to existing corpus."""

        jobs = self.config["processors"].split(",")
        out_obj = out_object_stanza(self.doc, jobs, start=0)
        print(out_obj.attrnames)
        out = out_obj.assemble_output_sent()

        ptags = ptags or out_obj.get_ptags()
        stags = out_obj.get_stags()
        # write out to .vrt
        outfile = mydict["advanced_options"]["output_dir"] + mydict["corpus_name"]

        out_object_stanza.write_vrt(outfile, out)
        if not add:
            encode_obj = be.encode_corpus(mydict)
            encode_obj.encode_vrt(ptags, stags)
        elif add:
            encode_obj = be.encode_corpus(mydict)
            encode_obj.add_tags_to_corpus(mydict, ptags, stags)


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

    def __init__(self, doc, jobs: list, start: int = 0):
        super().__init__(doc, jobs, start)
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
