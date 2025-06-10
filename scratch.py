import PartOne as po
import nltk
from nltk.corpus import cmudict

# df = po.read_novels()
# text2 = df.iloc[0,0]


text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"


# Download CMU Dictionary
nltk.download("cmudict")

cmu_dict = cmudict.dict()

# Extract the list of the corresponding word
w = 'unbelievable'

syl = po.count_syl(w, cmu_dict)
    
print(f"Word: {w}, Syllables: {syl}")