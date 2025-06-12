import nltk
import spacy
from pathlib import Path
import pandas as pd
import os
import string
import re
import nltk
import contractions as c
import pickle
import PartOne as po

df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")

for doc in df['parsed']:
    print(doc)