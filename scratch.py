import PartOne as po
import contractions as c


test_text = "This is MY StrING of TEXT!! REmoves number like 1206-9 and non-letter objects"
#    expect = ['this', 'is', 'my', 'string', 'of', 'text', 'removes', 'number', 'like', 'and', 'non', 'letter', 'objects']

test_text = "Does it Remove \n\n these types of \t\v special characters?"
    # expect = ['does','it','remove','these','types','of','special','characters']
    # assert po.tokens_clean(test_text) == ex
print(po.tokens_clean(test_text))# == expect