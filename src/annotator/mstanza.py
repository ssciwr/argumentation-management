# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be

stanza_dict = {
    "lang": "en",  # Language code for the language to build the Pipeline in
    "dir": r"/home/inga/stanza_resources",  # directory where models are stored
    "package": "default",  # see other models:
    # https://stanfordnlp.github.io/stanza/models.html
    "processors": "tokenize,pos",  # Comma-separated list of processors to use
    # can also be given as a dictionary:
    # {'tokenize': 'ewt', 'pos': 'ewt'}
    "logging_level": "INFO",  # DEBUG, INFO, WARN, ERROR, CRITICAL, FATAL
    # FATAL has least amount of log info printed
    "verbose": True,  # corresponds to INFO, False corresponds to ERROR
    "use_GPU": False,  # use GPU if available, False forces CPU only
    # kwargs for the individual processors
    "tokenize_model_path": r"./en/tokenize/combined.pt",
    # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
    # "mwt_model_path": r"./fr/mwt/gsd.pt",
    "pos_model_path": r"./en/pos/combined.pt",
    # "pos_pretrain_path": r"pretrain/gsd.pt",
    "tokenize_pretokenized": True,  # Use pretokenized text as input and disable tokenization
}


def fix_dict_path(dict):
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


def preprocess():
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
        self.nlp = sa.Pipeline(
            **self.config
        )  # Initialize the pipeline using a configuration dict

    def process_text(self, text):
        self.doc = self.nlp(text)  # Run the pipeline on the pretokenized input text
        return self.doc  # stanza prints result as dictionary

    def process_multiple_texts(self, textlist):
        # Wrap each document with a stanza.Document object
        in_docs = [sa.Document([], text=d) for d in textlist]
        self.mdocs = self.nlp(
            in_docs
        )  # Call the neural pipeline on this list of documents
        return self.mdocs


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
    mydict = fix_dict_path(stanza_dict)
    mytext = be.get_sample_text()
    # initialize instance of the class
    obj = mstanza_pipeline(mydict)
    obj.init_pipeline()
    out = obj.process_text(mytext)
    print(out)
