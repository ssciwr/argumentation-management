import spacy as sp

nlp = sp.load("en_core_web_sm")
doc = nlp("This is a test.")

print([i for i in doc.sents])

nlp = sp.load("en_core_web_md")
doc = nlp("This is a test 2.")
print([i for i in doc.sents])
