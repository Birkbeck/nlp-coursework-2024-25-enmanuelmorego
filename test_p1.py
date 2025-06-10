'''
Script to test that the functions works as expected
'''
import PartOne as po

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