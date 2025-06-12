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
        'text': ['the happy cat felt happiness in the delicious meal, but the happier dog looked at it with the most jealous eyes',
                 'The broccoli was soft, green and hot, the soft carrot was also hot, but the potatoes were the hottest',
                 'the tall, and very fast athlete went running in the rain']}

df = pd.DataFrame(data)
# Add parsed column 
parsed_list = []
for index, row in df.iterrows():
    parsed_obj = nlp(row['text'])
    parsed_list.append(parsed_obj)
    
df['parsed'] = parsed_list

expect = [('happy', 2), ('delicious', 1), ('jealous', 1), ('soft', 2), ('green', 1), ('hot', 3), ('tall', 1), ('fast', 1)]

print(po.adjective_counts(df))