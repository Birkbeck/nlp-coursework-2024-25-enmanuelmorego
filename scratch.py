import PartOne as po
import contractions as c


df = po.read_novels()
# text2 = df.iloc[0,0]
# #print(df.iloc[0,1])


t = po.get_fks(df)


print("="*80)
for key, value in t.items():
    print(f"* Title          : {key}\n* FKS Grade Level: {value}")
    print("-"*50)
#print(t2)

# String of contractions
test_text = "it's don't won't aren't we're we've i'd let's"
expect = ['is', 'is', 'do', 'not', 'will', 'not', 'are', 'not', 'we', 'are', 'we', 'have', 'I', 'would', 'let', 'us']
#print( po.tokens_clean(test_text))
print(c.fix(test_text))