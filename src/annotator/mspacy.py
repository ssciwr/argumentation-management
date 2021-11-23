import spacy as sp
from spacy.language import Language
from collections import defaultdict

# maybe for parallelising the pipeline, this should be done
# in base though
import os

cores = len(os.sched_getaffinity(0))


def preprocess(spec_pipe=None):
    """Load model"""

    # Add possibility for user to provide their own spacy model or
    # use specific one?
    if spec_pipe:
        nlp = sp.load(spec_pipe)

    else:
        # just this one model for now, what models are needed? There
        # are a lot of different ones in the old preprocessing
        print("Loading model en_core_web_sm")
        nlp = sp.load("en_core_web_sm")

    return nlp


def custom_pipeline():
    # Information about pipeline and the components:
    # https://spacy.io/usage/processing-pipelines

    config = {
        "[nlp]": {
            "lang": "en",
            "pipeline": ["tagger", "parser", "ner"],
            "before_creation": "null",
            "after_creation": "null",
            "after_pipeline_creation": "null",
            "batch_size": 1000,
        }
    }
    with open("test_conifg_spacy.cfg", "w") as configfile:
        for key in config:
            configfile.write(key + "\n")
            for item in config[key]:
                print(item)
                configfile.write("{} = {}\n".format(item, config[key][item]))

    # nlp = Language.from_config("./test_config_spacy.cfg")
    nlp = Language.from_config(config)
    return nlp


def apply_pipe(data, nlp):
    """Apply given pipeline to data"""
    # for multiprocessing or if we have multiple documents we can consider
    # something like:
    # docs = [doc for doc in nlp.pipe(data, n_process=cores)]
    # this should stream the individual documents and create Docs in order while
    # working in parallel. Maybe break down large corpus into peaces for this?
    return nlp(data)


# I guess we want to eiter load a pipeline or build one and apply it
# so we can assume that doc exists and if NER is called, that
# the pipeline included 'ner'
def NER(doc):
    """Get the NER tokens from Doc"""

    # store the named entities and associated information about label
    named_entities = defaultdict(list)
    # extract the information such as Label (as text and IOB code)
    # and position in text (by start/end char and start/end text ID)
    # the ID should correspond to the id in XML i think
    for ent in doc.ents:
        named_entities["Text: {} (IOB: {})".format(ent.text, ent.label)].append(
            {
                "Label_text": ent.label_,
                "IOB": ent.label,
                "Chars": [ent.start_char, ent.end_char],
                "Token Ids": [ent.start, ent.end],
            }
        )

    return named_entities


if __name__ == "__main__":
    with open("../../data/Original/iued_test_original.txt", "r") as file:
        data = file.read().replace("\n", "")
    doc = apply_pipe(data, preprocess())
    # for token in doc:
    #   print(token.text)
    named_ents = NER(doc)
    for key in named_ents:
        print("{} {}".format(key, named_ents[key]))

    # doesnt work atm
    doc2 = apply_pipe(data, custom_pipeline())
