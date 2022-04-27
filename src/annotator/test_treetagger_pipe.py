import base as be
import mtreetagger as mtt

if __name__ == "__main__":
    data = be.prepare_run.get_text("src/annotator/test/test_files/example_en.txt")
    mydict = be.prepare_run.load_input_dict("src/annotator/input_local")

    treetagger_dict = mydict["treetagger_dict"]
    pipe = mtt.treetagger_pipe(treetagger_dict)

    out = pipe.apply_to(data)

    print(out.doc)
    out.pass_results(mydict, "DICT")

    # be.decode_corpus(mydict).decode_to_file("out/")
