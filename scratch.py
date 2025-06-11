import PartOne as po
import nltk


df = po.read_novels()
# text2 = df.iloc[0,0]
# #print(df.iloc[0,1])


t = po.get_fks(df)


print("="*80)
for key, value in t.items():
    print(f"* Key: {key}\n* Value: {value}")
    print("-"*50)
#print(t2)

