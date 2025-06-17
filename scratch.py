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
from collections import Counter

import spacy

# nlp = spacy.load("en_core_web_sm")

# # Create fake data to test function
# data = {'title': ['novel1', 'novel2', 'novel3'],
#         'text': ['the happy cat felt happiness in the delicious meal, but the happier dog looked at it with the most jealous eyes',
#                  'The broccoli was soft, green and hot, the soft carrot was also hot, but the potatoes were the hottest. The potatoes were having a run',
#                  'the tall, and very fast athlete went running in the rain. The kind was running']}
import PartTwo as pt

print(pt.read_political_speeches_csv())