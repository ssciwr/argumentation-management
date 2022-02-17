# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be


class mstanza_preprocess:
    """Preprocessing for stanza document annotation. Collection
    of preprocessing steps to be carried out initially."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def fix_dict_path(dict) -> dict:
        # brute force to get model paths
        for key, value in dict.items():
            if "model" in key.lower():
                # if there is a prepending ".", remove it
                # be careful not to remove the dot before the file ending
                if "." in value[0:2]:
                    value = value.replace(".", "", 1)
                # prepend a slash to make sure
                value = "/" + value
                # combine the path from dir with the one from model
                value = dict["dir"] + value
                dict.update({key: value})
                print(dict[key], " updated!")
        return dict


class mstanza_pipeline:
    """Stanza main processing class.

    Args:
       config (dictionary): The full input dictionary.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (dictionary): The output dictionary with annotated tokens.
    """

    def __init__(self, mydict: dict):
        # we need the full dict to get the parameters for encoding
        self.mydict = mydict
        # just extract the stanza specific config here, is also less work for the user.
        self.config = be.prepare_run.update_dict(mydict["stanza_dict"])
        # does the activate_procs routine actually do anything here? Pytests work with and without it.
        self.config = be.prepare_run.activate_procs(self.config, "stanza_")

    def init_pipeline(self):
        # Initialize the pipeline using a configuration dict
        self.nlp = sa.Pipeline(**self.config)

    def process_text(self, text: str) -> dict:
        self.doc = self.nlp(text)  # Run the pipeline on the pretokenized input text
        return self.doc  # stanza prints result as dictionary

    def process_multiple_texts(self, textlist: list) -> dict:
        # Wrap each document with a stanza.Document object
        in_docs = [sa.Document([], text=d) for d in textlist]
        self.mdocs = self.nlp(
            in_docs
        )  # Call the neural pipeline on this list of documents
        return self.mdocs

    def postprocess(self):
        # postprocess of the annotated dictionary
        # fout = be.out_object.open_outfile(dict["output"])
        # sentencize using generic base output object
        # next step would be mwt, which is only applicable for languages like German and French
        # seems not to be available in spacy, how is it handled in cwb?
        jobs = be.prepare_run.get_jobs(self.config)
        out = out_object_stanza.assemble_output_sent(self.doc, jobs, start=0)
        # write out to .vrt
        out_object_stanza.write_vrt(self.mydict["output"], out)
        # encode -> move this out of here
        be.encode_corpus.encode_vrt(self.mydict)


def ner(doc):
    named_entities = defaultdict(list)

    for ent in doc.ents:
        # add the entities, key value is start and end char as Token IDs in stanza seem to be restricted to
        # individual sentences
        named_entities["{} : {}".format(ent.start_char, ent.end_char)].append(
            [ent.text, ent.type]
        )

    return named_entities


# inherit the output class from base and add stanza-specific methods
class out_object_stanza(be.out_object):
    """Out object for stanza annotation, adds stanza-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs, start):
        super().__init__(doc, jobs, start)
        self.attrnames = self.attrnames["stanza_names"]

    # add new method for stanza iteration over tokens/words/ents
    def iterate(self, out, sent):
        for token, word in zip(getattr(sent, "tokens"), getattr(sent, "words")):
            if token.text != word.text:
                raise NotImplementedError(
                    "Multi-word expressions not available currently"
                )
            tid = token.id[0] + self.tstart
            # for ent in getattr(sent, "ents"):
            # print(ent)
            out, line = self.collect_results(token, tid, word, out)
            out.append(line + "\n")
        self.tstart = tid
        return out
