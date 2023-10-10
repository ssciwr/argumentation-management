#!/bin/sh
# install the latest spacy and stanza models for nlpannotator
# this installs all implemented languages
# the package will download these also on demand though

# spacy
python -m spacy download en_core_web_md 
python -m spacy download de_core_news_md
python -m spacy download fr_core_news_md
python -m spacy download it_core_news_md
python -m spacy download ja_core_news_md
python -m spacy download pt_core_news_md
python -m spacy download ru_core_news_md
python -m spacy download es_core_news_md
#
# stanza
python -c "import stanza; stanza.download(\"en\")"
sleep 15
python -c "import stanza; stanza.download(\"de\")"
sleep 15
python -c "import stanza; stanza.download(\"ar\")"
sleep 15
python -c "import stanza; stanza.download(\"be\")"
sleep 15
python -c "import stanza; stanza.download(\"fr\")"
sleep 15
python -c "import stanza; stanza.download(\"it\")"
sleep 15
python -c "import stanza; stanza.download(\"ja\")"
sleep 15
python -c "import stanza; stanza.download(\"pt\")"
sleep 15
python -c "import stanza; stanza.download(\"ru\")"
sleep 15
python -c "import stanza; stanza.download(\"es\")"
sleep 15
python -c "import stanza; stanza.download(\"uk\")"