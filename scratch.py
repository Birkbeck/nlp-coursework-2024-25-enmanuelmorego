import PartOne as po
import nltk
from nltk.corpus import cmudict

# df = po.read_novels()
# text2 = df.iloc[0,0]


# text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"


# # Download CMU Dictionary
nltk.download("cmudict")

cmu_dict = cmudict.dict()
# print(po.count_syl('tow-fold', cmu_dict))

# Extract the list of the corresponding word
# test_words = {"cat": 1,"apple": 2,"banana": 3,"computer": 3,"education": 4,"unbelievable": 5,"unintelligible": 6}

# for key, value in test_words.items():
#     print(f"Key {key}: Value: {value}")
#     print(f"Syl count: {po.count_syl(key, cmu_dict)}")
#     print("."*20)

w = 'beautiful'
v = ['a','e','i','o','u']
vowel_cons_count = 0
syl_count = 0

# for l in w:
#     if l in v:
#         vowel_cons_count += 1
#     else:
#         if vowel_cons_count > 0:
#             syl_count += 1
#             vowel_cons_count = 0
#         else:
#             vowel_cons_count = 0

# print(f"my count{syl_count}")

#print(po.fk_level("Wow! In 2025, Amelia's\n\n robot baked 32 delicious apple pies — can you believe it?", cmu_dict))
t = "Wow! In 2025, Amelia's\n\n robot baked 32 delicious apple pies — can you believe it?"
print(nltk.tokenize.sent_tokenize(t))

test_text = "Does it Remove \n\n these types of \t\v special characters?"
print(po.tokens_clean(test_text))