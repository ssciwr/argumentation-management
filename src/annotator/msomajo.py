from somajo import SoMaJo
import annotator.base as be
import re


def purge(tokenized: str) -> str:
    """Function to search and replace problematic patterns in tokens
    after pretokenization."""

    # expand these with more if neccessary, correct mapping is important!
    patterns = ["ä", "ü", "ö", "ß", " "]
    solutions = ["ae", "ue", "oe", "ss", ""]

    for pattern, solution in zip(patterns, solutions):
        tokenized = re.sub("[{}]".format(pattern), solution, tokenized)

    return tokenized


def tokenize(text: list or str, model: str, split_sentences=True) -> str:
    """Function to tokenize text using somajo.tokenize_text. Text may be provided as
    string or as list of strings.

    [Args]:
           text[list[str] or str]: List of strings (paragraphs) or string.
           model[str]: Model to be used by somajo, options are de_CMC, en_PTB.
           split_sentences[bool]: Perform sentence splitting in addition to tokenization."""

    if type(text) == str:
        text = [text]

    tokenized = list(
        list(
            SoMaJo(
                model, split_camel_case=True, split_sentences=split_sentences
            ).tokenize_text(text)
        )
    )

    out = ""

    for sent in tokenized:
        out += "<s>\n"
        for token in sent:
            out += token.text + "\n"
        out += "</s>\n"

    out = purge(out)

    return out


def pretokenize(text: list or str, model: str, mydict: dict, split_sentences=True):
    """Function to run the pretokenize process with somajo and encode to corpus."""

    tokenized = tokenize(text, model, split_sentences=split_sentences)

    be.out_object.write_vrt(mydict["output"], tokenized)
    be.encode_corpus.encode_vrt(mydict)
