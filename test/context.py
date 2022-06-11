import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import annotator
from annotator import base
from annotator import pipe
from annotator import to_xml
from annotator import mspacy
from annotator import mstanza
from annotator import mtreetagger
from annotator import mflair
from annotator import msomajo
