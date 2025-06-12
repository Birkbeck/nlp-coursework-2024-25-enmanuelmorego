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
data = {'title'