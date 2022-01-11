# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be


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


def preprocess() -> object:
    """Download an English model into the default directory."""
    # this needs to be moved outside of the module and inside the docker container /
    # requirements - have a quick check here that file is there anyways
    print("Downloading English model...")
    sa.download("en")
    print("Downloading French model...")
    sa.download("fr")
    print("Building an English pipeline...")
    en_nlp = sa.Pipeline("en")
    return en_nlp


class mstanza_pipeline:
    """Stanza main processing class.

    Args:
       config (dictionary): The input dictionary with the stanza options.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (dictionary): The output dictionary with annotated tokens.
    """

    def __init__(self, config):
        self.config = config

    def init_pipeline(self):
        # Initialize the pipeline using a configuration dict
        self.nlp = sa.Pipeline(**self.config)

    def process_text(self, text) -> dict:
        self.doc = self.nlp(text)  # Run the pipeline on the pretokenized input text
        return self.doc  # stanza prints result as dictionary

    def process_multiple_texts(self, textlist) -> dict:
        # Wrap each document with a stanza.Document object
        in_docs = [sa.Document([], text=d) for d in textlist]
        self.mdocs = self.nlp(
            in_docs
        )  # Call the neural pipeline on this list of documents
        return self.mdocs

    def postprocess(self) -> str:
        # postprocess of the annotated dictionary
        # fout = be.out_object.open_outfile(dict["output"])
        # sentencize using generic base output object
        # next step would be mwt, which is only applicable for languages like German and French
        # seems not to be available in spacy, how is it handled in cwb?
        jobs = [proc.strip() for proc in mydict["processors"].split(",")]
        out = be.out_object.assemble_output_sent(
            self.doc,
            dict["output"],
            jobs,
            start=0,
            tool="stanza",
        )
        # write out to .vrt
        be.out_object.to_vrt(dict["output"], out)


def NER(doc):
    named_entities = defaultdict(list)

    for ent in doc.ents:
        # add the entities, key value is start and end char as Token IDs in stanza seem to be restricted to
        # individual sentences
        named_entities["{} : {}".format(ent.start_char, ent.end_char)].append(
            [ent.text, ent.type]
        )

    return named_entities


if __name__ == "__main__":
    dict = be.load_input_dict("input")
    # take only the part of dict pertaining to stanza
    stanza_dict = dict["stanza_dict"]
    # to point to user-defined model directories
    # stanza does not accommodate fully at the moment
    mydict = fix_dict_path(stanza_dict)
    # stanza does not care about the extra comment keys
    # but we remove them for subsequent processing just in case
    # now we need to select the processors and "activate" the sub-dictionaries
    mydict = be.update_dict(mydict)
    mydict = be.activate_procs(mydict, "stanza_")
    mytext = be.get_sample_text()
    # or use something shorter
    # mytext = "This is a test sentence for stanza. This is another sentence."
    # initialize instance of the class
    obj = mstanza_pipeline(mydict)
    obj.init_pipeline()
    out = obj.process_text(mytext)
    obj.postprocess()
    # For the output:
    # We need a module that transforms a generic dict into xml.
