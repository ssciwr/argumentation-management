import base as be


class SetConfig:
    """Sets the options in the config dictionaries for each tool.


    Here we set the pipelines depending on:
    1. processing option;
    2. processing type;
    3. text type and language (-> model selection);
    4. manually selected tool.
    """

    def __init__(self, mydict: dict) -> None:
        # select method
        self.processing_option = {
            "fast": self.pipe_fast,
            "accurate": self.pipe_accurate,
            "manual": self.pipe_manual,
        }
        self.accurate_dict = {
            "sentencize": "spacy",
            "tokenize": "spacy",
            "pos": "stanza",
            "lemma": "stanza",
            "ner": "stanza",
        }
        self.case = self.processing_option[mydict["processing_option"]]
        print("Selected option {}".format(self.case))
        self.case(mydict)

    def pipe_fast(self, mydict: dict):
        """Fast pipeline for efficient processing. Uses SpaCy."""
        self.tool = "spacy"
        self.processors = mydict["processing_type"]

    def pipe_accurate(self, mydict):
        """Accurate pipeline for accurate processing. Uses SpaCy and
        Stanza."""
        # find out which processors are being used
        # and order according to ordered dict
        self._get_processors(mydict["processing_type"])
        self._get_tools()

    def pipe_manual(self, mydict):
        pass

    def _get_processors(self, processors: list):
        # here we want to make sure the list of processors is clean and in correct order
        # separate the processor list at the comma
        processors = processors.split(",")
        processors = [i.strip() for i in processors]
        # order the processors based on the order dictionary
        order = {"sentencize": 0, "tokenize": 1, "pos": 2, "lemma": 3, "ner": 4}
        templist = [order.get(x) for x in processors]
        # make sure this stops if key is not in ordered dictionary
        if None in templist:
            print("Processing option not found!")
            print("Needs to be one of {}".format(order.keys()))
            raise (ValueError("You provided {}".format(processors)))
        ziplist = zip(templist, processors)
        ordlist = [x for _, x in sorted(ziplist)]
        self.processors = []
        for component in ordlist:
            self.processors.append(component)

    def _get_tools(self):
        self.tool = []
        for component in self.processors:
            self.tool.append(self.accurate_dict[component])
        print("added tool {} for components {}".format(self.tool, self.processors))

    @staticmethod
    def set_processors(dict_in: dict) -> dict:
        """Update the processor and language settings in the tool sub-dict.

        Args:
                mydict[dict]: Dict containing parameters."""
        # here we need to more generally map the pipeline
        # add in the defaults for each tool

        mytool = dict_in["tool"]
        mydict = dict_in
        mydict[mytool + "_dict"]["processors"] = mydict["processing_type"]
        mydict[mytool + "_dict"]["lang"] = mydict["language"]
        mydict[mytool + "_dict"]["text_type"] = mydict["document_type"]
        return mydict


if __name__ == "__main__":
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    print(mydict["processing_option"])
    mydict["processing_option"] = "accurate"
    mydict["processing_type"] = "pos  ,lemma, tokenize"
    obj = SetConfig(mydict)
