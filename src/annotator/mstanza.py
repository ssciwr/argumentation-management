# the stanza class object for stanza nlp
import stanza as sa


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


def pipeline_with_dict():
    # options for setting the paths to trained models do not work as they should
    config = {
        "lang": "fr",  # Language code for the language to build the Pipeline in
        "dir": "/home/inga/stanza_resources",  # directory where models are stored
        "package": "default",  # see other models:
        # https://stanfordnlp.github.io/stanza/models.html
        "processors": "tokenize,mwt,pos",  # Comma-separated list of processors to use
        # can also be given as a dictionary:
        # {'tokenize': 'ewt', 'pos': 'ewt'}
        "logging_level": "INFO",  # DEBUG, INFO, WARN, ERROR, CRITICAL, FATAL
        # FATAL has least amount of log info printed
        "verbose": True,  # corresponds to INFO, False corresponds to ERROR
        "use_GPU": False,  # use GPU if available, False forces CPU only
        # kwargs for the individual processors
        "tokenize_model_path": "/home/inga/stanza_resources/fr/tokenize/gsd.pt",  # Processor-specific arguments are set with keys "{processor_name}_{argument_name}"
        "mwt_model_path": "/home/inga/stanza_resources/fr/mwt/gsd.pt",
        "pos_model_path": "/home/inga/stanza_resources/fr/pos/gsd.pt",
        "pos_pretrain_path": "/home/inga/stanza_resources/fr/pretrain/gsd.pt",
        "tokenize_pretokenized": True,  # Use pretokenized text as input and disable tokenization
    }
    nlp = sa.Pipeline(**config)  # Initialize the pipeline using a configuration dict
    doc = nlp(
        "Van Gogh grandit au sein d'une famille de l'ancienne bourgeoisie ."
    )  # Run the pipeline on the pretokenized input text
    print(doc)  # Look at the result
    # stanza prints result as dictionary


def multiple_docs():
    nlp = sa.Pipeline(lang="en")  # Initialize the default English pipeline
    documents = [
        "This is a test document.",
        "I wrote another document for fun.",
    ]  # Documents that we are going to process
    in_docs = [
        sa.Document([], text=d) for d in documents
    ]  # Wrap each document with a stanza.Document object
    out_docs = nlp(in_docs)  # Call the neural pipeline on this list of documents
    print(
        out_docs[1]
    )  # The output is also a list of stanza.Document objects, each output corresponding to an input Document object


if __name__ == "__main__":
    # pipe = preprocess()
    pipeline_with_dict()
    # multiple_docs()
