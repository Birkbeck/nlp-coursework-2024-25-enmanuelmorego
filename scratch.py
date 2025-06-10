import PartOne as po


df = po.read_novels()
text2 = df.iloc[0,0]


# text = "This is the test string. ThIS contains NUMBERS like 123,123,4321. Also puncutation like €#,! Is numbERS"

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
test_text = """Dr. Lee arrived at 3 p.m.\nHe said, "Let's begin the test — quickly!"\tWhy? Because time's short...\vAnyway, the robot (Model X-2025) passed all 3 phases. Incredible, right? Let's do it again."""

t = po.get_ttrs(df)
print("="*80)
for key, value in t.items():
    print(f"{key:<50} {value}")
# t = "Wow! In 2025, Amelia's\n\n robot baked 32 delicious apple pies — can you believe it?"
# print(nltk.tokenize.sent_tokenize(t))

# test_text = "Does it Remove \n\n these types of \t\v special characters?"
# print(po.tokens_clean(test_text))