import pytest
import annotator.mtreetagger as mtt


@pytest.fixture
def read_test_en():
    with open("test/test_files/example_en_treetagger.vrt", "r") as f:
        data = f.read()
    return data


@pytest.fixture
def read_test_de():
    with open("test/test_files/example_de_treetagger.vrt", "r") as f:
        data = f.read()
    return data


def test_tokenize(read_test_de, read_test_en):

    text_de = "Das Feld der Computerchemie erfreut sich seit einiger Zeit einer stetig zunehmenden Aufmerksamkeit. Hierbei ist es für die Berechnung von chemisch relevanten Systemen von Interesse, möglichst große Systeme (Moleküle) durch effiziente Nutzung von Rechen- und Speicherressourcen zugänglich zu machen. Die Herausforderung besteht darin, dass mit der zunehmenden Anzahl von Atomen in größeren Systemen auch die Anzahl der internen Freiheitsgrade zunimmt."
    data_de = read_test_de

    assert mtt.tokenize(text_de, "de")[0] == data_de

    text_en = "The Scientific Software Center strives to improve scientific software development to ensure reproducible science and software sustainability. The SSC also acts as a link between the different scientific disciplines, enabling collaboration and interdisciplinary research. The current role of software in research communities Software development is an increasingly vital part of research, but if not done sustainably the result is often unmaintainable software and irreproducible science. This is due to a lack of software engineering training for scientists, limited funding for maintaining existing software and few permanent software developer positions. The SSC addresses the current shortcomings by implementing the three pillars of Development, Teaching and Outreach."
    data_en = read_test_en

    assert mtt.tokenize(text_en, "en")[0] == data_en
