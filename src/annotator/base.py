# the base class and utilities are contained in this module
import json


# read the sample data
def get_sample_text():
    name = "data/Original/iued_test_original.txt"
    with open(name, "r") as myfile:
        data = myfile.read().replace("\n", "")
    return data


# load the dictionary
def load_input_dict():
    with open("src/annotator/input.json") as f:
        dict = json.load(f)
    return dict


# open outfile
def open_outfile():
    name = "out/output.txt"
    f = open(name, "w")
    return f
