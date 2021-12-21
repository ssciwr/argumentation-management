import stanza
import spacy
import time

stanza.download("en")
time.sleep(15)
stanza.download("de")
spacy.load("en_core_web_sm")
