# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be

stanza_dict = {
    "lang": "en",  # Language code for the language to build the Pipeline in
    "dir": r"/home/inga/stanza_resources",  # directory where models are stored
    "package": "default",  # see other models:
    # https://stanfordnlp.github.io/stanza/models.html
    "processors": "tokenize,pos,lemma",  # Comma-separated list of processors to use
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
    "pos_batch_size": 3,  # this specifies the maximum number of words to process as
    #                      a minibatch for efficient processing. Default: 5000
    # "pos_pretrain_path": r"pretrain/gsd.pt",
    "lemma_model_path": r"./en/lemma/combined.pt",
    # lemma has several sub-options as per https://stanfordnlp.github.io/stanza/lemma.html
    # "tokenize_pretokenized": True,  # Use pretokenized text as input and disable tokenization
}
# Issues:
# only give model paths for the processors that are actually used, otherwise torch will throw
# exception


# Possible processors:
# 'tokenize': Tokenizes the text and performs sentence segmentation.
#        Dependency: -
# 'mwt': Expands multi-word tokens (MWT) predicted by the TokenizeProcessor.
#        This is only applicable to some languages.
#        Dependency: - 'tokenize'
# 'pos': Labels tokens with their universal POS (UPOS) tags, treebank-specific POS (XPOS) tags,
#        and universal morphological features (UFeats).
#        Dependency: - 'tokenize, mwt'
# 'lemma': Generates the word lemmas for all words in the Document.
#        Dependency: - 'tokenize, mwt, pos'
# 'depparse': Provides an accurate syntactic dependency parsing analysis.
#        Dependency: - 'tokenize, mwt, pos, lemma'
# 'ner': Recognize named entities for all token spans in the corpus.
#        Dependency: - 'tokenize, mwt'
# 'sentiment': Assign per-sentence sentiment scores.
#        Dependency: - 'tokenize, mwt'
# 'constituency': Parse each sentence in a document using a phrase structure parser.
#        Dependency: - 'tokenize, mwt, pos'


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
        # Initialize the pipeline using a configuration dict
        self.nlp = sa.Pipeline(**self.config)

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
    mytext = "This is a test sentence for stanza. This is another sentence."
    # initialize instance of the class
    obj = mstanza_pipeline(mydict)
    obj.init_pipeline()
    out = obj.process_text(mytext)
    # We need a module that transforms a generic dict into xml.
    # Have to decide on a structure though.
    # May be good to count tokens continuously and not starting from 1 every new
    # sentence.
    # print(out)
    # print sentences and tokens
    print([sentence.text for sentence in out.sentences])
    for i, sentence in enumerate(out.sentences):
        print(f"====== Sentence {i+1} tokens =======")
        print(
            *[f"id: {token.id}\ttext: {token.text}" for token in sentence.tokens],
            sep="\n",
        )
    # next step would be mwt, which is only applicable for languages like German and English
    #
    # print out pos tags
    print(
        *[
            f'word: {word.text}\tupos: {word.upos}\txpos: {word.xpos}\tfeats: \
        {word.feats if word.feats else "_"}'
            for sent in out.sentences
            for word in sent.words
        ],
        sep="\n",
    )
    #
    # access lemma
    print(
        *[
            f'word: {word.text+" "}\tlemma: {word.lemma}'
            for sent in out.sentences
            for word in sent.words
        ],
        sep="\n",
    )
