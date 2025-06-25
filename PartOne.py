#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
from spacy.symbols import nsubj, VERB
from pathlib import Path
import pandas as pd
import os
import string
import re
import contractions as c
import pickle
import math


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
    
    return fk_level


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

    '''
    Args: 
        doc: a selected column and row from a data frame that contains the SpaCy parsed doc
        target_verb: A verb to search in the data and find the corresponding syntatic subject
    
    Returns:
        A list of the top 10 most common syntatic subjects for a given verb
        ordered by their Pointwise Mutual Information (descending)
   
    Note:
        The function below calculates the Pointwise Mutual Information (PMI) between the top 10 syntatic subject of the verbs.
   
    ==IMPORTANT==: 
    The PMI was calculated instead of the Positive Pointwise Mutual Information (PPMI) as that was the promt specification
    However, usually PPMI is preferred as per the textbook
    The difference between the PPMI and Pointwise Mutual Information (PMI) is that PPMI takes the max between 0 and PMI
    Meaning, all negative PMI become 0
        - Why transform negative PMI to 0? 
            A negative PMI indicates that the word pair appeared less often than chance. However, not all words are equally common. This 
            poses the risk that less common word pairs might return a negative PMI not becuase they appeared less than chance, but simply 
            because not enough data was collected to accurately represent the less common words. 
    '''
    # Extract the top 10 subject-verb pairs 
    verb_subject = subjects_by_verb_count(doc, target_verb)
    # Extract unique words from the s-v pairs 
    unique_w = set()
    for d in verb_subject:
        for k in d.keys():
            for ind in k:
                unique_w.add(ind)

    # Initialise total words counter
    total_words = 0
    # Initialise counter for the words of the subject-verb pairs
    total_existing = {w: 0 for w in unique_w}
    # Count objects
    for token in doc:
        # Only include words, not punctuation marks, etc
        if token.text.isalpha():
            total_words += 1
            # Use lemmatized version of words
            word_lemma = token.lemma_
            if word_lemma in total_existing:
                total_existing[word_lemma] += 1
        
    # Probability of the verb (context)
    p_c = total_existing[target_verb]/total_words
    # Initialise dictionary for ppmi scores
    pmi_dict = {}
    # Loop over the dictionary of the 10 s-v pairs
    for d in verb_subject:
        for key, value in d.items():
            # probability of the pair verb-subject
            p_wc = value/total_words

            # Probability of the subject (word)
            p_w = total_existing[key[1]]/total_words

            # Calculate PMI
            ## Only if p_w_ and p_c are not 0
            if p_c == 0 or p_w == 0:
                pmi = 0
            else:
                pmi = p_wc/(p_w * p_c)
                pmi = math.log2(pmi)
            # Add value to final dict
            pmi_dict[key] = pmi_dict.get(key, round(pmi,3))

    # Sort final dictionary
    pmi_dict = sorted(pmi_dict.items(), key = lambda item: item[1], reverse = True)
    #return pmi_dict
    '''Uncomment the code below to return a list of dictionaries containg the {(verb, sub): pmi_score...}'''
    #return [{vs_pair: count} for vs_pair, count in pmi_dict]
    '''Uncoment the code below to return a list of the most common subjects [word1, word2....]'''
    return [key[1] for key, value in pmi_dict]
    

def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""

    '''
    Args: 
        doc: a selected column and row from a data frame that contains the SpaCy parsed doc
        verb: A verb to search in the data and find the corresponding syntatic subject
    
    Returns:
        A list of dictionaries containing the verb-subject pair as key, and the frequency count as the value. 
        The list is sorted in descending order (more frequent counts first), and returns the first 10 available items in the list

        
    ================================================= IMPORTANT ============================================================
    The prompt of this question indicated that the function should "a list of the ten most common syntactic subjects 5 of the 
    verb ‘to hear’ (in any tense) in the text, ordered by their frequency".

    The function below returns the list as requested, but extra information is provided in the list. 
    This extra information allows the function to satisfy the promt whilst adhering to coding principles and best practices:

        - Following the coding principles of DRY (Dont repeat yourself), the function returns:
            - A list of 10 elements where each element is a dictionary
            - Each dictonary contains the verb, subject pair, and the frequency of the pair
                i.e.,: [{(verb, subj1): 7}, ..... {(verb, subj_x): 3}]

        The reason for including the extra informationis that the count of how many times the verb, subj pair appears 
        in a document is needed to calculate the PMI. 

        Therefore, instead of copying the same code in two functions (identify and count the times a given pair appears),
        we can simply pass this function in the PMI function to obtain those values. This means we can keep the code clean,
        avoid repeating code, and ensuring each function has its unique and single purpose in the program
    =========================================================================================================================
    '''
    
    '''
    The syntatic subject of a verb is the subject that performs the action that the verb refers to. 
    For example: 'The boy drank the water'. The verb is 'drank'. We can find the syntatic subject by asking
    who drank the water? the boy. Hence the syntatic subject in this case is 'the boy'

    Spacy provides head and child relationship to determine how the words are conected by the syntatic arch. Each
    word has a single corresponding head. 

    We can iterate over each token (word) and check the `dep` attribute (dependency)
        If the value is `nsubj` then the word is a nominal subject, which means that the token is functioning as the grammatical subject
        Then, because each token has a corresponding head, we extract the corresponding head_text and its head_pos. 
            If the head_pos is verb, then we have retrieved the Verb and the Syntatic Subject of the verb
    '''
    # Initialise the output dictionary
    verb_subj = {}
    sorted_dict = {}

    # Clean the users input to ensure is a valid verb
    target_verb_l = nlp(verb)
    # Only extract the verb from the string
    target_verb_l = ([token.lemma_ for token in target_verb_l if token.pos_ == 'VERB'])
    # Raise value error if user passes more than one verb
    if len(target_verb_l) > 1:
        raise ValueError("Please enter one verb only")
    # Raise value error if the user passes no verbs at all
    elif len(target_verb_l) < 1:
        raise ValueError("Please enter one verb")
    # Save the verb as a string only if passed correctly
    else:
        target_verb_l = target_verb_l[0]

    # Loop over the tokens in the parsed doc
    for token in doc:
        # Syntatic relationship of token is nominal subject, and its head is a verb
        if token.dep == nsubj and token.head.pos == VERB:
            # extract lemmatized versions of: verb (head) and nsubj (token),
            verb = token.head.lemma_
            subj = token.lemma_
            # only count for the desired verb
            if verb == target_verb_l:
                # count pair frequency
                verb_subj[(verb, subj)] = verb_subj.get((verb, subj), 0) + 1
    # Create dict with title and sort by value
    sorted_dict = sorted(verb_subj.items(), key = lambda item: item[1], reverse = True)

    # Filter to top 10 - if 10 matches are available
    if len(sorted_dict) > 10:
        sorted_dict = sorted_dict[:10]

    return [{vs_pair: count} for vs_pair, count in sorted_dict]
    

def adjective_counts(doc):
    ######                                                                                   ######
    ###### Please note, adjective counts were removed from the requirements of the assigment ###### 
    ######                                                                                   ######
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
    Function to count syllables based on vowels clusters. Included 'y' as vowels as many english words contain only 'y' i.e., 'why', 'by', 'cry', etc

    Args:
        word: a string representing a single word
    
    Returns:
        int: total number of syllables per word
    '''
    # Define vowels
    v = ['a','e','i','o','u', 'y']
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

def count_obj(doc):
    '''
    Function that counts the syntatic objects in a document
    
    Args:
        doc: A column of a dataframe containing Soacy Doc object
    
    Returns:
        A list of the top 10 most common objects, lemmatized, along with the count, presented as a dictionary per each item [{lemma_word1: count}, ..., {lemma_wordn: count}]
    '''
    # Inititalise dictionary
    syntatic_object = {}
    # Get tags to identify objects
    object_tags = ['dobj', 'iobj', 'oprd', 'obj', 'pobj']

    # Extract words
    for token in doc:
        # Extract the type of dependency, explained
        dep_tag = token.dep_
        if dep_tag in object_tags:
            # Lemmatize word
            word_lemma = token.lemma_
            syntatic_object[word_lemma] = syntatic_object.get(word_lemma, 0) + 1 
        
    # Sort dictionary
    sorted_dict = sorted(syntatic_object.items(), key = lambda item: item[1], reverse = True)

    # Filter to top 10 - if 10 matches are available
    if len(sorted_dict) > 10:
        sorted_dict = sorted_dict[:10]

    '''Uncomment the code below to return a list of dictionaries containg the {subj: count}...}'''
    #return [{vs_pair: count} for vs_pair, count in sorted_dict]
    '''Uncoment the code below to return a list of the most common subjects [word1, word2....]'''
    return [pair[0] for pair in sorted_dict]




if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    print(df.head())
    nltk.download("cmudict")
    parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    # Please note, adjective counts were removed from the requirements of the assigment
    print(adjective_counts(df))
    
    print("\n*** Top 10 Syntatic objects overall in the text - Organised by count ***\n")
    for i, row in df.iterrows():
        print(row['title'])
        print(count_obj(row['parsed']))
        print("\n")

    print("\n*** Top 10 Subjects by verb 'hear' - Organised by count ***\n")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")
    
    print("\n*** Top 10 Subjects by verb 'hear' - Organised by PMI ***\n")
    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")
    

