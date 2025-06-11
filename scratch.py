import PartOne as po
import contractions as c


# df = po.read_novels()
# # text2 = df.iloc[0,0]
# # #print(df.iloc[0,1])


# t = po.get_fks(df)


# print("="*80)
# for key, value in t.items():
#     print(f"* Title          : {key}\n* FKS Grade Level: {value}")
#     print("-"*50)
#print(t2)

# String of contractions
test_text = "d'Urbervilles I'd like to know the new plan! it's great but won't go for it. Lets stick to the old plan"

#print( po.tokens_clean(test_text))
#test_text = "You don't know what you were to me, once.  Why, once ... Oh, I can't bear to think of it!"
print(po.tokens_clean(test_text))
#print(c.fix(test_text))