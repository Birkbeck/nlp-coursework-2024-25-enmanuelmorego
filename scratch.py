import PartOne as po
import nltk
from nltk.corpus import cmudict

# df = po.read_novels()
# text2 = df.iloc[0,0]


text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"


# Download CMU Dictionary
nltk.download("cmudict")

cmu_dict = cmudict.dict()
po.count_syl('education', cmu_dict)

# Extract the list of the corresponding word
# test_words = {"cat": 1,"apple": 2,"banana": 3,"computer": 3,"education": 4,"unbelievable": 5,"unintelligible": 6}

# for key, value in test_words.items():
#     print(f"Key {key}: Value: {value}")
#     print(f"Syl count: {po.count_syl(key, cmu_dict)}")
#     print("."*20)
 