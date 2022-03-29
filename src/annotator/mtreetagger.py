import treetaggerwrapper as ttw
import annotator.base as be


def tokenize(text: str, lang: str) -> str:
    """Function to tokenize text using treetaggerwrapper.TreeTagger.tag_text. Text is provided as string.

    [Args]:
            text[str]: String of text to be tokenized.
            lang[str]: Two-char language code, i.e. "en" for english or "de" for german."""

    # load the tokenizer
    tokenizer = ttw.TreeTagger(TAGLANG=lang)
    # tokenize the text
    tokenized = [
        token
        for token in tokenizer.tag_text(
            text,
            prepronly=True,
            notagurl=True,
            notagemail=True,
            notagip=True,
            notagdns=True,
        )
        if token
    ]

    # convert to string in vrt format
    out = ""

    for token in tokenized:
        out += token + "\n"

    # replace problematic patterns
    out = be.out_object.purge(out)
    # text is not sentencized
    sentencized = False

    return out, sentencized
