import PartOne as po
import contractions as c

df = po.read_novels()
p_df = po.parse(df)

print(p_df.head())