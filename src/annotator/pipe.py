import base as be


class SetConfig:
    """Sets the options in the config dictionaries for each tool.


    Here we set the pipelines depending on:
    1. processing option;
    2. processing type;
    3. text type and language (-> model selection);
    4. manually selected tool.
    """

    def __init__(self) -> None:
        pass

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
        return mydict @ staticmethod
