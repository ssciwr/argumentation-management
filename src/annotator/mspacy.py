import spacy as sp
from spacy.lang.en import English
from collections import defaultdict

# maybe for parallelising the pipeline, this should be done
# in base though
import os

# cores = len(os.sched_getaffinity(0))


# initialize the spacy top level class, this currently only
# holds some parameters
class spaCy:
    def __init__(self, config, jobs, pretrained=False):

        # config to build a pipeline
        self.config = config
        #  list of which Results to fetch, ideally corresponding
        # to loaded components in config, currently only supports
        # "NER"
        self.jobs = jobs
        # use a full pretrained pipeline if desired, this will run
        # all components of said pipeline
        if pretrained:
            self.pretrained = pretrained
        else:
            self.pretrained = False


# build the pipeline from config-dict
class spaCy_pipe(spaCy):

    # init with specified config, this may be changed later?
    # -> Right now needs quite specific instuctions
    def __init__(self, config, jobs, pretrained=False):
        super().__init__(config, jobs, pretrained=pretrained)

        if self.pretrained:
            self.nlp = sp.load(self.pretrained)

        else:
            self.nlp = sp.blank(self.config["lang"])
            for factory in self.config["add_to"]:
                try:
                    self.nlp.add_pipe(factory, source=sp.load(self.config["source"]))
                except OSError:
                    inp = input(
                        "Could not find {} on system. Attempt to download? [Y/N]".format(
                            self.config["source"]
                        )
                    )

                    if inp == "Y":
                        os.system(
                            "python -m spacy download {}".format(self.config["source"])
                        )
                        self.nlp.add_pipe(
                            factory, source=sp.load(self.config["source"])
                        )
                    elif inp == "N":
                        exit()

    # call the build pipeline on the data
    def apply_to(self, data):
        self.doc = self.nlp(data)
        return self

    def grab_NER(self):

        # store the named entities and associated information about label
        self.named_entities = dict()
        # extract the information such as Label (as text and IOB code)
        # and position in text (by start/end char and start/end text ID)
        # the ID should correspond to the id in XML i think
        for ent in self.doc.ents:
            # if entity is one token we just append it
            if ent.end - ent.start == 1:
                self.named_entities["{}".format(ent.start)] = {
                    "Label_text": ent.label_,
                    "IOB": ent.label,
                    "Chars": [ent.start_char, ent.end_char],
                    "Token Ids": [ent.start, ent.end],
                    "Text": ent.text,
                }
            # if entity is more than one token we also add the Ids of
            # all contained tokens to the dict and assign them the same labels etc.
            else:
                for i in range(ent.start, ent.end):
                    self.named_entities["{}".format(i)] = {
                        "Label_text": ent.label_,
                        "IOB": ent.label,
                        "Chars": [ent.start_char, ent.end_char],
                        "Token Ids": [ent.start, ent.end],
                        "Text": ent.text,
                    }

        return self.named_entities

    def grab_results(self):

        # grab the results after running the pipeline, currently only for NER
        results = dict()
        if "NER" in self.jobs:
            results["NER"] = self.grab_NER()

        return results


if __name__ == "__main__":
    with open("../../data/Original/iued_test_original.txt", "r") as file:
        data = file.read().replace("\n", "")

    config = {"lang": "en", "add_to": ["ner"], "source": "en_core_web_sm"}
    jobs = ["NER"]

    # either load pipe with config from above or load a complete pretrained pipe
    # this will load all components
    results = spaCy_pipe(config, jobs).apply_to(data).grab_results()
    # , pretrained="en_core_web_sm")

    # print(results)
    for key, value in results["NER"].items():
        print(
            "ID {} : [IOB]= {} [Text]={}, [Label]={}".format(
                key, value["IOB"], value["Text"], value["Label_text"]
            )
        )

    print("*" * 50)

    different_config = {"lang": "en", "add_to": ["ner"], "source": "en_core_web_md"}

    results1 = spaCy_pipe(different_config, jobs).apply_to(data).grab_results()

    # print(results1)

    for key, value in results1["NER"].items():
        print(
            "ID {} : [IOB]= {} [Text]={}, [Label]={}".format(
                key, value["IOB"], value["Text"], value["Label_text"]
            )
        )


'''

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


def pipeline_with_dict():

    # Information about pipeline and the components:
    # https://spacy.io/usage/processing-pipelines

    config = {
        "name":"en_core_web_sm", #name of model
        "disable":[], # components to be disabled, given as list, loaded but not used
        "exclude":["tagger","parser","attribute_ruler","lemmatizer"], # components to be excluded, given as list, wont be loaded
        "config":{}
    }# put the instructions for sp.laod() into dict

    nlp = sp.load(**config)
    return nlp


def blank_with_dict():
    # load a default language and then add components from a pretrained pipeline for
    # that language
    config = {
        "lang":"en",
        "add_to":["tok2vec", "tagger","parser","ner","attribute_ruler","lemmatizer"],
        "source":"en_core_web_sm"
    }# config equivalent to sp.load("en_core_web_sm")


    configNER = {
        "lang":"en",
        "add_to":["ner"],
        "source":"en_core_web_sm"
    }# load just NER from en_core_web_sm

    nlp = sp.blank(configNER["lang"])
    for factory in configNER["add_to"]:
        nlp.add_pipe(factory, source=sp.load(configNER["source"]))

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
        named_entities["Start Id: {}".format(ent.start)].append(
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

    # use the regular model
    doc = apply_pipe(data, preprocess())
    # for token in doc:
    #   print(token.text)
    named_ents = NER(doc)
    for key in named_ents:
        print("{} {}".format(key, named_ents[key]))

    # also use regular model but exclude components
    doc2 = apply_pipe(data, pipeline_with_dict())
    print('*'*20)
    # get NER
    named_ents2 = NER(doc2)
    # check that NER still works
    for key in named_ents:
        print("{} {}".format(key, named_ents2[key]))
        if key in named_ents2:
            try:
                assert named_ents2[key] == named_ents[key]
            except AssertionError:
                print(named_ents[key], named_ents2[key])
        else:
            print("!Untracked token in ents2: {}".format(named_ents[key]))

    # "rebuilt" en_core_web_sm manually from basic sp.lang.en.English
    doc3 = apply_pipe(data, blank_with_dict())
    print('*'*20)
    named_ents3 = NER(doc3)
    # check that NER still works
    for key in named_ents3:
        print("{} {}".format(key, named_ents3[key]))
        if key in named_ents3:
            try:
                assert named_ents3[key] == named_ents[key]
            except AssertionError:
                print(named_ents[key], named_ents3[key])
        else:
            print("!Untracked token in ents3: {}".format(named_ents[key]))
    # -> NER still works fine, but labels one entity differently for some reason

'''
