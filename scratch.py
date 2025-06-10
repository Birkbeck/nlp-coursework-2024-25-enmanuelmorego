import PartOne as po

df = po.read_novels()
text2 = df.iloc[0,0]


text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"



print(po.nltk_ttr(text))

for index, row in df.iterrows():
    print(f"Row {index:<2}: {row['title']:<30} ({row['year']}) Type-Token Ratio {po.nltk_ttr(row['text']):.4f}")
print(":"*80)
t = po.get_ttrs(df)
for key, value in t.items():
    print(f"{key:<10}: {value:<25}")