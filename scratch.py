import PartOne as po

df = po.read_novels()
text2 = df.iloc[0,0]


text = "This is a test, with punctuation! And Mixed CASE."

print(po.nltk_ttr(text))