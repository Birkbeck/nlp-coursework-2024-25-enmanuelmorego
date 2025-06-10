import PartOne as po
import nltk


df = po.read_novels()
text2 = df.iloc[0,0]
print(df.iloc[0,1])


# text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"
cmudict = nltk.corpus.cmudict.dict()
t2 = po.get_fks(df)

text = "The artist is the creator of beautiful things.  To reveal art and conceal the artist is art's aim.  The critic is he who can translate into another manner or a new material his impression of beautiful things."

t = po.fk_level(text, cmudict)


print("="*80)
print(f"Dorian Grey Extract: Readbility raw {t}") 
print(t2)
