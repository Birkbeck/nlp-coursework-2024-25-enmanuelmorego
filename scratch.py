import PartOne as po
import contractions as c

df = po.read_novels()
df_sub = (df.iloc[0:5])
p_df = po.parse(df_sub)


print(p_df.head())