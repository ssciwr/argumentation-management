import treetaggerwrapper as ttw
import annotator.base as be


def tokenize(text: str, lang: str) -> str:
    """Function to tokenize text using treetaggerwrapper.TreeTagger.tag_text. Text is provided as string.

    [Args]:
            text[str]: String of text to be tokenized.
            lang[str]: Two-char language code, i.e. "en" for english or "de" for german."""

    tokenizer = ttw.TreeTagger(TAGLANG=lang)
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

    out = ""

    for token in tokenized:
        out += token + "\n"

    out = be.out_object.purge(out)
    sentencized = False

    return out, sentencized
