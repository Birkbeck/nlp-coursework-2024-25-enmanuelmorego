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

data = {'title': ['novel1', 'novel2', 'novel3'],
        'text': ['the boy ran. The boy runs very fast. The kids were running',
                 'the girl was running. I also run. I ran home. They ran very far',
                 'the dog barked']}
df = pd.DataFrame(data)

# # Load spaCy model
nlp = spacy.load("en_core_web_sm")

# # Add new column with spaCy Doc objects
df['parsed'] = df['text'].apply(nlp)

df_mini = df.iloc[0]
# count of co occurrence
verb_subject = po.subjects_by_verb_count(df_mini['parsed'], 'run')
print('\n')
print(f"1. Counts of Verbs and Subjects:\n\t{verb_subject}\n")

# Extract unique objects
unique_w = set()
for d in verb_subject:
    for k in d.keys():
        for ind in k:
            unique_w.add(ind)

print(f"2. Extract unique words for total counts: \n\t{unique_w}\n")

# Count word occurrence in whole document, and total tokens in doc
total_words = 0
total_existing = {}
for token in df_mini['parsed']:
    # Ensure only words are counted
    if token.is_lpha():
        print(token)
