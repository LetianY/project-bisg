import os
import pandas as pd
from src.constants import constants as const


make_counts = False
path = "../data/raw/census"
file_name = "Names_2010Census.csv"
file_path = os.path.join(path, file_name)
assert os.path.exists(file_path), "Set the path and file_name variables"

#TODO: Revisit the meaning of the "(S)" here
na_values = ['None', '(S)', 'S']
df = pd.read_csv(file_path, na_values=na_values)


#Drop the rows we don't need
df = df.drop(["rank", "prop100k", "cum_prop100k"], axis=1)
df = df.rename(columns={
    "pctwhite": const.white,
    "pctblack": const.black,
    "pctapi": const.api,
    "pctaian": const.aian,
    "pct2prace": const.multiple,
    "pcthispanic": const.hisp
})


save_path = "../data/clean/census"
save_name = "surnames.csv"

if make_counts:
    save_name = "surnames_counts.csv"
    df.iloc[:,2:] = df.iloc[:,2:].astype(float).mul(df[const.count], axis=0)

df.iloc[:, 2:] = df.iloc[:, 2:].astype(float) / 100
"""
The last row contains some "all other names" row which is 
kinda good info, but we don't need to save it in our data cleaning

There's also a lot of nan values where the census ppl were unsure 
of the true propositions, so we're also going to fill those 
nans with 0. 
"""
df = df.iloc[:-1].fillna(0)

#ReOrder Columns
df.iloc[:,2:] = df[const.races]
df.columns = df.columns[:2].to_list() + const.races

df.to_csv(os.path.join(save_path, save_name), index=False)


