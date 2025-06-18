'''
Script to test that the functions works as expected
'''
import pytest
import PartOne as po
import nltk
from nltk.corpus import cmudict
import spacy
import pandas as pd

def test_token_clean1():
    '''
    Function that cleans and tokenize a string of text using word_tokenizer
    '''
    test_text = "This is MY StrING of TEXT!! REmoves number like 1206-9 and non-letter objects"
    expect = ['this', 'is', 'my', 'string', 'of', 'text', 'removes', 'number', 'like', 'and', 'nonletter', 'objects']
    assert po.tokens_clean(test_text) == expect

    test_text = "Does it Remove \n\n these types of \t\v special characters?"
    expect = ['does','it','remove','these','types','of','special','characters']
    assert po.tokens_clean(test_text) == expect

def test_token_clean2():
    '''
    Test that function handles contractions as expected
    '''
    # String of contractions
    test_text = "it's don't won't aren't we're we've i'd let's"
    expect = ['it', 'is', 'do', 'not', 'will', 'not', 'are', 'not', 'we', 'are', 'we', 'have', 'i', 'would', 'let', 'us']
    assert po.tokens_clean(test_text) == expect

    # Mix of words 
    test_text = "d'Urbervilles I'd like to know the new plan! it's great but won't go for it. Lets stick to the old plan"
    expect = ["durbervilles","i","would","like","to","know","the","new","plan","it","is","great","but","will","not","go","for","it","let","us","stick","to","the","old","plan"]
    assert po.tokens_clean(test_text) == expect


def test_nltk_ttr1():
    '''
    Test that the function recieves a text object with:
        Letters, number and punctuation marks
        Upper and lower case
    Removes non-letter items, transforms all to lower case and calculates the ttr of the cleaned tokens
    '''
    test_text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"
    # Text above contains 14 tokens and  10 unique tokens (removing number and puncptuation and ignoring case. Expect TTR = 0.7143)
    assert po.nltk_ttr(test_text) == 0.7143

def test_count_syl1():
    '''
    Test to ensure syllable counter works as expected
    '''
    # Download CMU Dictionary
    nltk.download("cmudict")
    cmu_dict = cmudict.dict()

    test_words = {"cat": 1,"aPPle": 2,"banana": 3,"computEr": 3,"education": 4,"unbelievable": 5,"unintelligible": 6, "there":1}

    for key, value in test_words.items():
        assert po.count_syl(key, cmu_dict) == value

def test_count_syl_vowel_cluster1():
    '''
    Test that vowel cluster count work as expected
    '''
    assert po.count_syl_vowel_cluster('beautiful') == 3
    assert po.count_syl_vowel_cluster('zoo') == 1
    assert po.count_syl_vowel_cluster('audio') == 2
    assert po.count_syl_vowel_cluster('temperature') == 5

def tests_fk_level1():
    '''
    Test that fklevel functions works as expected
    '''
    text = """On an evening in the latter part of May a middle-aged man was walking homeward from Shaston to the village of Marlott, in the adjoining Vale of Blakemore, or Blackmoor. 
            The pair of legs that carried him were rickety, and there was a bias in his gait which inclined him somewhat to the left of a straight line.  
            He occasionally gave a smart nod, as if in confirmation of some opinion, though he was not thinking of anything in particular.  
            An empty egg-basket was slung upon his arm, the nap of his hat was ruffled, a patch being quite worn away at its brim where his thumb came in taking it off.
            Presently he was met by an elderly parson astride on a gray mare, who, as he rode, hummed a wandering tune."""
    cmudict = nltk.corpus.cmudict.dict()
    t = po.fk_level(text, cmudict)
    assert round(t, 4) == 11.1088

def test_adjective_counts1():
    '''
    Test that adjective counts function works properly
    '''

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

    assert po.adjective_counts(df) == expect

def test_subjects_by_verb1():
    '''
    Test that function handles bad input as expected
    '''
    nlp = spacy.load("en_core_web_sm")
    doc = "Sally hears you. Sally can hear you. Sally wants to hear you. To hear him sing is a joy for Sally"
    doc_p = nlp(doc)

    # Passing multiple verbs
    with pytest.raises(ValueError):
        po.subjects_by_verb_count(doc_p, "to hear running testing")

    # Passing non verbs
    with pytest.raises(ValueError):
        po.subjects_by_verb_count(doc_p, "not a verb")

def test_subjects_by_verb2():
    '''
    test function with valid output
    '''
    # Set data
    data = {'title': ['novel1', 'novel2', 'novel3'],
            'text': ['the boy ran. The boy runs very fast. The kids were running',
                    'the girl was running. I also run. I ran home. They ran very far',
                    'the dog barked']}
    df = pd.DataFrame(data)

    # Load spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Add new column with spaCy Doc objects
    df['parsed'] = df['text'].apply(nlp)

    # Check that each subdict also is the correct length
    expect_len = [2,3,0]
    i = 0
    for i, row in df.iterrows():
        assert len(po.subjects_by_verb_count(row["parsed"], "run")) == expect_len[i]
        i += 1

def test_count_obj1():
    '''
    Tests that the syntatic object counter works as expected
    '''
    # Define dummy text
    text = 'the cat chased the mouse and caught the mouse, but the dog chased the cat and found it in the garden.'
    # Transform into SpaCy doc
    nlp = spacy.load("en_core_web_sm")
    text_nlp = nlp(text)
    # Test function
    count_dict = po.count_obj(text_nlp)

    assert count_dict == [{'mouse': 2}, {'cat': 1}, {'it': 1}, {'garden': 1}]




