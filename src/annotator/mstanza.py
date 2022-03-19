# the stanza class object for stanza nlp
from collections import defaultdict
import stanza as sa
import base as be


class Stanza:
    """Stanza main processing class.

    Args:
       config (dictionary): The full input dictionary.
       text (string): The raw text that is to be processed.
       text (list of strings): Several raw texts to be processed simultaneously.
       annotated (dictionary): The output dictionary with annotated tokens.
    """

    def __init__(self, config: dict):
        # we need the full dict to get the parameters for encoding
        self.config = config
        # Initialize the pipeline using a configuration dict
        self.nlp = sa.Pipeline(**self.config)

    @staticmethod
    def fix_dict_path(dict: dict) -> dict:
        """Function to update the model path in given input dict.

        Args:
                dict[dict]: Dictionary containing path to be updated."""

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

    def apply_to(self, text: str) -> dict:
        """Funtion to apply pipeline to provided textual data.

        Args:
                text[str]: Textual Data as string."""

        self.doc = self.nlp(text)  # Run the pipeline on the pretokenized input text
        return self

    def process_multiple_texts(self, textlist: list) -> dict:
        """Function to process multiple texts.

        Args:
                textlist[list]: List containing the texts."""

        # Wrap each document with a stanza.Document object
        in_docs = [sa.Document([], text=d) for d in textlist]
        self.mdocs = self.nlp(
            in_docs
        )  # Call the neural pipeline on this list of documents
        return self.mdocs

    def pass_results(self, out_param: dict) -> None:
        """Funtion to write post-pipeline data to .vrt file and encode for CWB.

        Args:
                out_param[dict]: Parameters for output."""

        jobs = be.prepare_run.get_jobs(self.config)
        out = out_object_stanza.assemble_output_sent(self.doc, jobs, start=0)
        # write out to .vrt
        outfile = out_param["advanced_options"]["output_dir"] + out_param["corpus_name"]
        out_object_stanza.write_vrt(outfile, out)
        be.encode_corpus.encode_vrt(out_param)


def ner(doc) -> dict:
    """Function to extract NER tags from Doc Object."""

    named_entities = defaultdict(list)

    for ent in doc.ents:
        # add the entities, key value is start and end char as Token IDs in stanza seem to be restricted to
        # individual sentences
        named_entities["{} : {}".format(ent.start_char, ent.end_char)].append(
            [ent.text, ent.type]
        )

    return named_entities


# inherit the output class from base and add stanza-specific methods
class out_object_stanza(be.out_object):
    """Out object for stanza annotation, adds stanza-specific methods to the
    vrt/xml writing."""

    def __init__(self, doc, jobs: list, start: int):
        super().__init__(doc, jobs, start)
        self.attrnames = self.attrnames["stanza_names"]

    # add new method for stanza iteration over tokens/words/ents
    def iterate(self, out: list, sent, style: str) -> list:
        """Function to iterate through sentence object and extract data to list.

        Args:
                out[list]: List containing the collected output.
                sent[stanza sent-Object]: Object containing annotated sentence."""

        for token, word in zip(getattr(sent, "tokens"), getattr(sent, "words")):
            if token.text != word.text:
                raise NotImplementedError(
                    "Multi-word expressions not available currently"
                )
            tid = token.id[0] + self.tstart
            line = self.collect_results(token, tid, word, style)

            out.append(line + "\n")
        self.tstart = tid
        return out


if __name__ == "__main__":
    # read in input.json
    mydict = be.prepare_run.load_input_dict("./src/annotator/input")
    mydict["tool"] = "stanza"
    mydict["input"] = "./src/annotator/test/test_files/example_de.txt"
    stanza_dict = mydict["stanza_dict"]
    stanza_dict = be.prepare_run.update_dict(stanza_dict)
    stanza_dict["lang"] = "de"
    stanza_dict["processors"] = "tokenize,pos,mwt,lemma"
    stanza_dict["dir"] = "./src/annotator/test/models/"
    stanza_dict = be.prepare_run.activate_procs(stanza_dict, "stanza_")
    data = be.prepare_run.get_text(mydict["input"])
    # initialize the pipeline with the dict
    stanza_pipe = Stanza(stanza_dict)
    # apply pipeline to data, and encode
    stanza_pipe.apply_to(data).pass_results(mydict)
