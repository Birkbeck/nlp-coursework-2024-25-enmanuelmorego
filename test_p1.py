'''
Script to test that the functions works as expected
'''
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
