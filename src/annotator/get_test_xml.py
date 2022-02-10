import to_xml as txml
import base as be
import mstanza as ma


mydict = {
    "lang": "en",
    "dir": "./test/models/",
    "processors": "tokenize,pos,lemma",
}
obj = ma.mstanza_pipeline(mydict)
obj.init_pipeline()

with open("./test/test_files/example_en.txt") as f:
    text = f.read().replace("\n", "")

doc = obj.process_text(text)

data = doc.to_dict()

print(type(data))

raw_xml = txml.Element("doc")

sents = [txml.list_to_xml("Sent", i, elem) for i, elem in enumerate(data, 1)]

for sent in sents:
    raw_xml.append(sent)

# for i, elem in enumerate(data, 1):
#     raw_xml.append(txml.list_to_xml("Sent", i, elem))

raw_xml = txml.to_string(raw_xml)

xml = txml.beautify(raw_xml)

with open("example_en.xml", "w") as file:
    file.write(xml)
