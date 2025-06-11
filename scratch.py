import PartOne as po
import nltk


# df = po.read_novels()
# text2 = df.iloc[0,0]
# #print(df.iloc[0,1])


# text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"
cmudict = nltk.corpus.cmudict.dict()
#t2 = po.get_fks(df)

text = """On an evening in the latter part of May a middle-aged man was walking homeward from Shaston to the village of Marlott, in the adjoining Vale of Blakemore, or Blackmoor. 
The pair of legs that carried him were rickety, and there was a bias in his gait which inclined him somewhat to the left of a straight line.  
He occasionally gave a smart nod, as if in confirmation of some opinion, though he was not thinking of anything in particular.  
An empty egg-basket was slung upon his arm, the nap of his hat was ruffled, a patch being quite worn away at its brim where his thumb came in taking it off.
Presently he was met by an elderly parson astride on a gray mare, who, as he rode, hummed a wandering tune."""
print("."*80)
print(text)
#text = "there"
t = po.fk_level(text, cmudict)


print("="*80)
print(round(t, 4))
#print(t2)

