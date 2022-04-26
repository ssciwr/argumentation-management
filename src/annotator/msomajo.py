from somajo import SoMaJo
import base as be


def tokenize(text: list or str, model: str, split_sentences=True) -> str and bool:
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

    out = be.out_object.purge(out)
    sentencized = True

    return out, sentencized
