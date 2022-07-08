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
            "sentencize": "somajo",
            "tokenize": "somajo",
            "pos": "stanza",
            "lemma": "stanza",
            "ner": "stanza",
        }
        self.map_processors = {
            "spacy": {
                "sentencize": "senter, parser",
                "tokenize": "tok2vec",
                "lemma": "lemmatizer",
                "pos": "tagger",
                "ner": "ner",
            },
            "stanza": {
                "sentencize": "tokenize",
                "tokenize": "tokenize",
                "lemma": "lemma",
                "pos": "pos",
                "mwt": "mwt",
            },
            "somajo": {
                "sentencize": "sentencize",
                "tokenize": "tokenize",
            },
            "treetagger": {"tokenize": "tokenize", "pos": "pos", "lemma": "lemma"},
            "flair": {"pos": "pos", "ner": "ner"},
        }
        self.mydict = mydict
        self.case = self.processing_option[self.mydict.get("processing_option", "fast")]
        self.case()
        # validate that processors and tool have same length
        self._validate_processors()
        # select language and model
        if "spacy" in self.tool:
            self._set_model_spacy()
        if "stanza" in self.tool:
            self._set_model_stanza()
        if "somajo" in self.tool:
            self._set_model_somajo()
        if "treetagger" in self.tool:
            self._set_model_treetagger()
        if "flair" in self.tool:
            self._set_model_flair()
        self.set_processors()
        self.set_tool()

    def _pipe_fast(self):
        """Fast pipeline for efficient processing. Uses SpaCy."""
        print("Selected fast pipeline.")
        # make sure processors are ordered and correct
        processors = self._get_processors(self.mydict["processing_type"])
        self._order_processors(processors)
        # get length of processors and repeat spacy as many time for each
        # option
        self.tool = ["spacy" for _ in self.processors]

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
        self.tool = self._get_processors(self.mydict["tool"])
        if len(self.tool) == 1 and len(self.processors) != 1:
            self.tool = [self.tool[0] for _ in self.processors]
        print(self.tool, self.processors)

    def _get_processors(self, processors: str) -> list:
        # here we want to make sure the list of processors is clean and in correct order
        # separate the processor list at the comma
        if "," in processors:
            processors = processors.split(",")
            processors = [i.strip() for i in processors]
        else:
            processors = [processors]
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

    def _validate_processors(self):
        if len(self.processors) != len(self.tool):
            raise ValueError("Selected tools do not match selected processors!")

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
            self.mydict["spacy_dict"]["model"] = self.mydict["model"]
            print("Using selected model {}.".format(self.mydict["model"]))
        else:
            # now we check selected language to
            # choose adequate model
            if self.mydict["language"] == "en":
                self.mydict["spacy_dict"]["model"] = "en_core_web_md"
            elif self.mydict["language"] == "de":
                self.mydict["spacy_dict"]["model"] = "de_core_news_md"
            elif self.mydict["language"] == "fr":
                self.mydict["spacy_dict"]["model"] = "fr_core_news_md"
            elif self.mydict["language"] == "it":
                self.mydict["spacy_dict"]["model"] = "it_core_news_md"
            elif self.mydict["language"] == "ja":
                self.mydict["spacy_dict"]["model"] = "ja_core_news_md"
            elif self.mydict["language"] == "pt":
                self.mydict["spacy_dict"]["model"] = "pt_core_news_md"
            elif self.mydict["language"] == "ru":
                self.mydict["spacy_dict"]["model"] = "ru_core_news_md"
            elif self.mydict["language"] == "es":
                self.mydict["spacy_dict"]["model"] = "es_core_news_md"
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
        print("Setting language options for Stanza.")
        print(
            """If you require a model other than the default for the
        specified language, you will need to set it manually."""
        )
        # see here for a selection of models:
        # https://stanfordnlp.github.io/stanza/available_models.html
        if "model" in self.mydict:
            self.mydict["stanza_dict"]["model"] = self.mydict["model"]
            print("Using selected model {}.".format(self.mydict["model"]))
        else:
            # select based on language
            self.mydict["stanza_dict"]["model"] = None

    def _set_model_somajo(self):
        """Update the model depending on language - somajo."""
        print("Setting language options for SoMaJo.")
        if self.mydict["language"] == "en":
            self.mydict["somajo_dict"]["model"] = "en_PTB"
        elif self.mydict["language"] == "de":
            self.mydict["somajo_dict"]["model"] = "de_CMC"
        else:
            raise ValueError(
                "Language {} not available for SoMaJo. Aborting ...".format(
                    self.mydict["language"]
                )
            )

    def _set_model_treetagger(self):
        """Update the model depending on language and text option - treetagger."""
        print("Setting model and language options for Treetagger.")
        languages_token_pos_lemma = ["en", "de", "fr", "es"]
        languages_pos_lemma = [
            "bg",
            "nl",
            "et",
            "fi",
            "gl",
            "it",
            "kr",
            "la",
            "mn",
            "pl",
            "ru",
            "sk'",
            "sw",
        ]
        if self.mydict["language"] in languages_pos_lemma:
            # check that tokenization is not selected for these languages
            for proc, mytool in zip(self.processors, self.tool):
                if mytool == "treetagger" and proc == "tokenize":
                    print(
                        "Tokenization only possible for languages {}".format(
                            languages_token_pos_lemma
                        )
                    )
                    raise ValueError(
                        "Tokenization not available in treetagger for the selected language {}.".format(
                            self.mydict["language"]
                        )
                    )
        self.mydict["treetagger_dict"]["lang"] = self.mydict["language"]

    def _set_model_flair(self):
        """Update the model depending on selected processors - flair."""
        print("Language and processing options for flair are set at flair runtime.")
        print("This ensures correct tagger is loaded.")
        self.mydict["flair_dict"]["model"] = []
        if self.mydict["language"] == "en":
            pos_string = "pos"
            ner_string = "ner"
        elif self.mydict["language"] == "de":
            pos_string = "de-pos"
            ner_string = "de-ner"
        else:
            raise ValueError(
                "Currently only en and de models for flair! You selected language {}".format(
                    self.mydict["language"]
                )
            )
        for proc, mytool in zip(self.processors, self.tool):
            if mytool == "flair" and proc == "pos":
                self.mydict["flair_dict"]["model"].append(pos_string)
            elif mytool == "flair" and proc == "ner":
                self.mydict["flair_dict"]["model"].append(ner_string)
        if len(self.mydict["flair_dict"]["model"]) == 1:
            # flair does only need list for more than one processor
            self.mydict["flair_dict"]["model"] = self.mydict["flair_dict"]["model"][0]

    def set_processors(self) -> dict:
        """Update the processor and language settings in the tool sub-dict."""
        # in some cases, there may be duplicates in the processor list
        # remove the duplicates - TODO
        # using set reorders the processors but we only now of duplicates
        # after the ordering
        # first purge already existing processors in tooldict
        for mytool in set(self.tool):
            self.mydict[mytool + "_dict"]["processors"] = []
        for proc, mytool in zip(self.processors, self.tool):
            # map to new name
            myname = self.map_processors[mytool][proc]
            print("found name {} for tool {} and proc {}".format(myname, mytool, proc))
            for i in myname.split(", "):
                self.mydict[mytool + "_dict"]["processors"].append(i)
            # we don't need the language for spacy
            self.mydict[mytool + "_dict"]["lang"] = self.mydict["language"]
            # if self.model:
            # self.mydict[mytool + "_dict"]["model"] = self.model
        # For stanza, processors must be comma-separated string
        # (no spaces)
        temp = ",".join(self.mydict["stanza_dict"]["processors"])
        self.mydict["stanza_dict"]["processors"] = temp
        # update processors in dict
        self.mydict["processing_type"] = self.processors
        return self.mydict

    def set_tool(self) -> dict:
        # update tool in dict
        self.mydict["tool"] = self.tool
        return self.mydict
