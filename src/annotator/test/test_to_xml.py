import pytest
import to_xml as txml
import mstanza as ma


@pytest.fixture()
def get_data():

    mydict = {
        "lang": "en",
        "dir": "./test/models/",
        "processors": "tokenize,pos,lemma",
    }
    obj = ma.Stanza(mydict)

    with open("./test/test_files/example_en.txt") as f:
        text = f.read().replace("\n", "")

    doc = obj.apply_to(text)

    return doc.to_dict()


def test_list_to_xml(get_data):
    """Function to test list to xml transformation."""

    sent = get_data[0]

    test_xml = txml.list_to_xml("Sent", 1, sent)

    assert (
        txml.to_string(test_xml)
        == '<Sent Id="1"><Token Id="1"><text>The</text><lemma>the</lemma><upos>DET</upos><xpos>DT</xpos><feats>Definite=Def|PronType=Art</feats><start_char>0</start_char><end_char>3</end_char></Token><Token Id="2"><text>Scientific</text><lemma>Scientific</lemma><upos>ADJ</upos><xpos>NNP</xpos><feats>Degree=Pos</feats><start_char>4</start_char><end_char>14</end_char></Token><Token Id="3"><text>Software</text><lemma>Software</lemma><upos>PROPN</upos><xpos>NNP</xpos><feats>Number=Sing</feats><start_char>15</start_char><end_char>23</end_char></Token><Token Id="4"><text>Center</text><lemma>Center</lemma><upos>PROPN</upos><xpos>NNP</xpos><feats>Number=Sing</feats><start_char>24</start_char><end_char>30</end_char></Token><Token Id="5"><text>strives</text><lemma>strive</lemma><upos>VERB</upos><xpos>VBZ</xpos><feats>Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin</feats><start_char>31</start_char><end_char>38</end_char></Token><Token Id="6"><text>to</text><lemma>to</lemma><upos>PART</upos><xpos>TO</xpos><start_char>39</start_char><end_char>41</end_char></Token><Token Id="7"><text>improve</text><lemma>improve</lemma><upos>VERB</upos><xpos>VB</xpos><feats>VerbForm=Inf</feats><start_char>42</start_char><end_char>49</end_char></Token><Token Id="8"><text>scientific</text><lemma>scientific</lemma><upos>ADJ</upos><xpos>JJ</xpos><feats>Degree=Pos</feats><start_char>50</start_char><end_char>60</end_char></Token><Token Id="9"><text>software</text><lemma>software</lemma><upos>NOUN</upos><xpos>NN</xpos><feats>Number=Sing</feats><start_char>61</start_char><end_char>69</end_char></Token><Token Id="10"><text>development</text><lemma>development</lemma><upos>NOUN</upos><xpos>NN</xpos><feats>Number=Sing</feats><start_char>70</start_char><end_char>81</end_char></Token><Token Id="11"><text>to</text><lemma>to</lemma><upos>PART</upos><xpos>TO</xpos><start_char>82</start_char><end_char>84</end_char></Token><Token Id="12"><text>ensure</text><lemma>ensure</lemma><upos>VERB</upos><xpos>VB</xpos><feats>VerbForm=Inf</feats><start_char>85</start_char><end_char>91</end_char></Token><Token Id="13"><text>reproducible</text><lemma>reproducible</lemma><upos>ADJ</upos><xpos>JJ</xpos><feats>Degree=Pos</feats><start_char>92</start_char><end_char>104</end_char></Token><Token Id="14"><text>science</text><lemma>science</lemma><upos>NOUN</upos><xpos>NN</xpos><feats>Number=Sing</feats><start_char>105</start_char><end_char>112</end_char></Token><Token Id="15"><text>and</text><lemma>and</lemma><upos>CCONJ</upos><xpos>CC</xpos><start_char>113</start_char><end_char>116</end_char></Token><Token Id="16"><text>software</text><lemma>software</lemma><upos>NOUN</upos><xpos>NN</xpos><feats>Number=Sing</feats><start_char>117</start_char><end_char>125</end_char></Token><Token Id="17"><text>sustainability</text><lemma>sustainability</lemma><upos>NOUN</upos><xpos>NN</xpos><feats>Number=Sing</feats><start_char>126</start_char><end_char>140</end_char></Token><Token Id="18"><text>.</text><lemma>.</lemma><upos>PUNCT</upos><xpos>.</xpos><start_char>140</start_char><end_char>141</end_char></Token></Sent>'
    )


def test_dict_to_xml(get_data):
    """Function to test dict to xml transformation."""

    token = get_data[0][0]

    test_xml = txml.dict_to_xml("Token", token)

    assert (
        txml.to_string(test_xml)
        == "<Token><id>1</id><text>The</text><lemma>the</lemma><upos>DET</upos><xpos>DT</xpos><feats>Definite=Def|PronType=Art</feats><start_char>0</start_char><end_char>3</end_char></Token>"
    )


def test_beautify(get_data):
    """Test the beautifier."""

    data = get_data

    raw_xml = txml.Element("doc")

    for i, elem in enumerate(data, 1):
        raw_xml.append(txml.list_to_xml("Sent", i, elem))

    raw_xml = txml.to_string(raw_xml)

    xml = txml.beautify(raw_xml)

    with open("./test/test_files/example_en.xml", "r") as f:
        check_xml = f.read()

    assert xml == check_xml
