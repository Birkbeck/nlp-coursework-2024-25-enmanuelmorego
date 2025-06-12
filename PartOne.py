#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

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

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000

# Download CMU Dictionary
nltk.download("cmudict")



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    # Count total words
    words = tokens_clean(text)
    total_words = len(words)
    # Count total sentences
    text_cleaned = re.sub(r'[\n\t\r\f\v]', ' ', text).strip()
    sentences = nltk.tokenize.sent_tokenize(text_cleaned)
    total_sentences = len(sentences)
    # Count total syllables
    total_syl = 0
    for w in words:
        total_syl += count_syl(w, d)

    fk_level = (0.39*(total_words/total_sentences)) + (11.8*(total_syl/total_words)) - 15.59
    # print(f"Words: {total_words}")
    # print(f"Sentences: {total_sentences}")
    # print(f"Syllables: {total_syl}")
    # print(f"FK Grade Level: {fk_level}")
    
    return fk_level
    # return {"Words": total_words,
    #         "Sentences": total_sentences,
    #         "Syllables": total_syl,
    #         "FK Grade Level": fk_level}



def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    # Initialise syllable counter
    syl_count = 0
    # Find the word in the dictionary
    w = d.get(word.lower(), False)
    if w:
        # Extract the first list of phoneme (if there are multiple pronunciations for words, there would be multiple list)
        # We are only interested on counting syllables so taking the first list works as CMU has the most common pronunciation as item 0
        syl_list = w[0]
        for syl in syl_list:
            # Syllables are labeled on objects that end with a number (representing the stress of pronunciation)
            if syl[-1].isdigit():
                syl_count += 1
    # TODO clarify vowel cluster assignment
    else:
       syl_count += count_syl_vowel_cluster(word)

    return syl_count

def read_novels(path=Path.cwd() / "texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year
    
    Args:
        Function defaults to a specific location to search for the files unless otherwise specified

    Returns: 
        A DataFrame with the text, title, author, and year
        The DataFrame is sorted by year with new indeces matching the sorting pattern
    """
    # Extract filenames
    files = os.listdir(path)
    # Initialise list of dictionaries
    data_dict = []

    for f in files:
        # Extract values for column headers
        title, author, year = f.split("-")
        # Join file name with directory path
        file_load = os.path.join(path, f)

        # Load data/text from documents
        with open(file_load, 'r', encoding='utf-8') as file:
            content = file.read()
            # Remove special break characters
            content = re.sub(r"\s+", " ", content).strip()

        # Add them to our data object
        data_dict.append({'text': content, 'title': title, 'author':author, 'year': year[:-4]})
        # Transform dictionary into a pandas data frame
        df = pd.DataFrame(data_dict)

    # Return sorted data frame with clean indeces
    return df.sort_values("year").reset_index(drop=True)
   


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    # Initialise empty list for parsed objects
    parsed_list = []
    # Get max lenght of nlp
    nlp_max = nlp.max_length
    timer = len(df)
    # Loop over data frame to parse texts
    for i, r, in df.iterrows():
        # Warn user if text lenght is greater than nlp max
        if len(r['text']) > nlp_max:
            print(f"\n**** WARNING ****\nLenght of document {r['title']} ({len(r['text'])}) is greater than SpaCy models' max ({nlp_max})\n- Please review...\n")
            # Append None in place
            parsed_list.append(None)
        else:
            # Parse and tokenize
            parsed_obj = nlp(r['text'])
            # Store parsed object to list
            parsed_list.append(parsed_obj)
        print("*"*timer)
        timer -= 1
    # Add parsed_list to dataframe
    df['parsed'] = parsed_list

    # Save document
    out_path = store_path/out_name
    with open(out_path, "wb") as file:
        pickle.dump(df, file)

    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    # Replace words separated by "-" with " " and transform text to lower case
    text = text.lower().replace("-"," ")
    # Tokenize document
    tokens = nltk.word_tokenize(text)
    # Remove punctuation marks (only keep alpha characters)
    tokens = [word for word in tokens if word.isalpha()]
    #tokens = tokens_clean(text) See first # TODO
    # Calculate type-token ratio
    ttr = round(len(set(tokens))/len(tokens),4)
    
    return ttr


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    pass



def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""

    # initiliase dictionary for frequency count
    adj_dict = {}
    # Loop thru each row of the data frame
    for i, row in doc.iterrows():
        # Extract the spacy doc (parsed column) into an object
        doc_object = row['parsed']
        # For each doc, iterate thru each token
        for token in doc_object:
            if token.pos_ == "ADJ":
                # Extract adjective
                adj = token.lemma_
                adj_dict[adj] = adj_dict.get(adj, 0) + 1
    # Conver dict into list of tuples
    return list(adj_dict.items())


def tokens_clean(text):
    # TODO maybe this is not needed
    '''
    Function that takes a given text and cleans it by:
        - Making all text lower case
        - Removing punctuation and numbers
    
        Args:
            text: a string object
        
        Returns:
            list: a list of cleaned tokens
    '''
    # Clean contractions
    text = c.fix(text)
    # Split tokens into a list
    tokens = text.split()
    # Make all tokens lower case
    tokens = [t.lower() for t in tokens]
    # Identify all punctionation marks
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # Remove punctuation
    tokens = [re_punc.sub('', token) for token in tokens]
    # Only keep alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    return tokens

def count_syl_vowel_cluster(word):
    '''
    Function to count syllables based on vowels clusters

    Args:
        word: a string representing a single word
    
    Returns:
        int: total number of syllables per word
    '''
    # Define vowels
    v = ['a','e','i','o','u']
    cons_vowel = 0
    syl_count = 0
    word = word.lower()

    # Loop over word
    for l in word:
        if l in v:
            cons_vowel += 1
        # If letter is not a vowel
        else:
            # If there are previous vowels:
            if cons_vowel > 0:
                # Add a syllable count
                syl_count += 1
                # Reset consecutive counter
                cons_vowel = 0
            # If not greater than 0, go to next iteration
    # Capture cases where the last iteration is a sequence of vowels
    if cons_vowel > 0:
        syl_count += 1

    return syl_count



if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    #parse(df)
    #print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    # print(adjective_counts(df))
    """ 
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    """

