from flair.data import Sentence
from flair.models import SequenceTagger
from flair.tokenization import SegtokSentenceSplitter
from collections import defaultdict


class Flair:
    def __init__(self, lang):

        self.lang = lang

        # load in a tagger based on the job I guess


class Flair_NER(Flair):

    # initialize flair NER by loading the tagger specified by lang
    def __init__(self, lang):
        super().__init__(lang)
        if self.lang == "de":
            self.tagger = SequenceTagger.load("de-ner")
        elif self.lang == "en":
            self.tagger = SequenceTagger.load("ner")

    # splitter = SegtokSentenceSplitter()

    def __call__(self, tokens):
        sentences = [Sentence(tokens)]
        self.tagger.predict(sentences)

        self.named_entities = defaultdict(list)

        for sentence in sentences:
            for entity in sentence.get_spans():
                self.named_entities[
                    "Start {} End {}".format(
                        entity.start_pos, str(entity.end_pos).split()[0]
                    )
                ].append([entity.text, entity.labels[0]])

        return self.named_entities


if __name__ == "__main__":
    FlairNER = Flair_NER("en")
    path = "../../data/Original/iued_test_original.txt"

    with open(path, "r") as file:
        data = file.read().replace("\n", "")

    named_ents = FlairNER(data)
    print(named_ents)
