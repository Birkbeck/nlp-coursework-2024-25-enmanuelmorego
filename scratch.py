import PartOne as po

df = po.read_novels()
text2 = df.iloc[0,0]


text = "Hello there! In 2023, NASA launched 4.5 rockets — that's 50% more than last year. Is this the future? Maybe. iPhone sales soared, tHe world watched."


print(po.nltk_ttr(text))

for index, row in df.iterrows():
    print(f"Row {index:<2}: {row['title']:<30} ({row['year']}) Type-Token Ratio {po.nltk_ttr(row['text']):.4f}")