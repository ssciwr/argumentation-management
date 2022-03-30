import base as be


class SetConfig:
    """Sets the options in the config dictionaries for each tool.


    Here we set the pipelines depending on:
    1. processing option and processing type;
    2. text type and language (-> model selection).
    """

    def __init__(self, mydict: dict) -> None:
        # select method
        self.processing_option = {
            "fast": self._pipe_fast,
            "accurate": self._pipe_accurate,
            "manual": self._pipe_manual,
        }
        self.accurate_dict = {
            "sentencize": "spacy",
            "tokenize": "spacy",
            "pos": "stanza",
            "lemma": "stanza",
            "ner": "stanza",
        }
        self.mydict = mydict
        self.case = self.processing_option[self.mydict["processing_option"]]
        self.case()
        # select language and model
        if "spacy" in self.tool:
            self._set_model_spacy()
        if "stanza" in self.tool:
            self._set_model_stanza()
        self.set_processors()

    def _pipe_fast(self):
        """Fast pipeline for efficient processing. Uses SpaCy."""
        print("Selected fast pipeline.")
        # make sure processors are ordered and correct
        processors = self._get_processors(self.mydict["processing_type"])
        self._order_processors(processors)
        self.tool = ["spacy"]

    def _pipe_accurate(self):
        """Accurate pipeline for accurate processing. Uses SpaCy and
        Stanza."""
        print("Selected accurate pipeline.")
        # find out which processors are being used
        # and order according to ordered dict
        processors = self._get_processors(self.mydict["processing_type"])
        self._order_processors(processors)
        self._get_tools()

    def _pipe_manual(self) -> None:
        """Manual selection of tools per processor type."""
        print("Selected manual pipeline.")
        # here we assume that the user set the processors in the correct order
        self.processors = self._get_processors(self.mydict["processing_type"])
        # convert to list and make sure the list of tools has no blanks
        # here we assume that the tools are are written correctly and exist
        self.tool = []
        self.tool.append(self._get_processors(self.mydict["tool"]))

    def _get_processors(self, processors: str) -> list:
        # here we want to make sure the list of processors is clean and in correct order
        # separate the processor list at the comma
        processors = processors.split(",")
        processors = [i.strip() for i in processors]
        return processors

    def _order_processors(self, processors: list) -> None:
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

    def _get_tools(self) -> None:
        self.tool = []
        for component in self.processors:
            self.tool.append(self.accurate_dict[component])
        print("added tool {} for components {}".format(self.tool, self.processors))

    def _set_model_spacy(self):
        """Update the model depending on language and text option - spacy."""
        print("Setting model and language options for SpaCy.")
        # check if a model was set manually - in this case we
        # do not want to overwrite
        if "model" in self.mydict:
            self.model = self.mydict["model"]
            print("Using selected model {}.".format(self.model))
        else:
            # now we check selected language and text type to
            # choose adequate model
            if self.mydict["language"] == "en":
                if self.mydict["document_type"] == "text":
                    # standard text
                    self.model = "en_core_web_md"
                elif self.mydict["document_type"] == "scientific":
                    # scientific text, use the scispacy package for biomedical text
                    self.model = "en_core_sci_md"
                elif self.mydict["document_type"] == "historic":
                    # historic, use model for old language
                    self.model = "en_core_web_md"
            elif self.mydict["language"] == "de":
                if self.type == "text":
                    # standard text
                    self.model = "de_core_news_md"
                elif self.mydict["document_type"] == "scientific":
                    # scientific text
                    self.model = "de_core_news_md"
                elif self.mydict["document_type"] == "historic":
                    # historic, use model for old language
                    self.model = "de_core_news_md"
            else:
                # make sure to throw an exception if language is not found
                raise ValueError(
                    """Language {} not available yet.
                Aborting...""".format(
                        self.mydict["language"]
                    )
                )

    def _set_model_stanza(self):
        """Update the model depending on language and text option - stanza."""
        print("Setting model and language options for Stanza.")
        # see here for a selection of models:
        # https://stanfordnlp.github.io/stanza/available_models.html
        # we only select based on language, not text type
        pass

    def set_processors(self) -> dict:
        """Update the processor and language settings in the tool sub-dict."""
        # here we need to more generally map the pipeline
        # and add in the defaults for each tool
        for i, mytool in enumerate(self.tool):
            if len(self.tool) == 1:
                self.mydict[mytool + "_dict"]["processors"] = self.processors
            elif len(self.tool) > 1:
                self.mydict[mytool + "_dict"]["processors"].append(self.processors[i])
            mydict[mytool + "_dict"]["lang"] = self.mydict["language"]
            if self.model:
                self.mydict[mytool + "_dict"]["model"] = self.model
        return self.mydict


if __name__ == "__main__":
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    mydict["processing_option"] = "accurate"
    mydict["processing_type"] = "pos  ,lemma, tokenize"
    obj = SetConfig(mydict)
    # now we still need to add the order of steps - processors was ordered list
    # need to access that and tools to call tools one by one
