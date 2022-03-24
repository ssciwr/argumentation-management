import annotator.msomajo as msomajo
import annotator.base as be

text = """The Scientific Software Center strives to improve scientific software development to ensure reproducible science and software sustainability. The SSC also acts as a link between the different scientific disciplines, enabling collaboration and interdisciplinary research. The current role of software in research communities Software development is an increasingly vital part of research, but if not done sustainably the result is often unmaintainable software and irreproducible science. This is due to a lack of software engineering training for scientists, limited funding for maintaining existing software and few permanent software developer positions. The SSC addresses the current shortcomings by implementing the three pillars of Development, Teaching and Outreach."""

if __name__ == "__main__":

    default_dict = be.prepare_run.load_input_dict("src/annotator/input_local")
    mydict = be.prepare_run.get_encoding(default_dict)

    msomajo.pretokenize(text, "en_PTB", mydict)

    be.decode_corpus(mydict).decode_to_file(directory="out", verbose=False)