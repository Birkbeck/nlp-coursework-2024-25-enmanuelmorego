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

import spacy

nlp = spacy.load("en_core_web_sm")

# Create fake data to test function
data = {'title': ['novel1', 'novel2', 'novel3'],
        'text': ['there was a pink happy cat',
                 'the broccoli was soft green and hot',
                 'the athlete went running']}

df = pd.DataFrame(data)
# Add parsed column 
parsed_list = []
for index, row in df.iterrows():
    parsed_obj = nlp(row['text'])
    parsed_list.append(parsed_obj)
    
df['parsed'] = parsed_list

print(po.adjective_counts(df))