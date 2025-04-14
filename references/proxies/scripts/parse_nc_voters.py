import os

import numpy as np
import pandas as pd
from src.constants import constants as const


path = "../data/raw/nc"
file_name = "ncvoter1.txt"
file_path = os.path.join(path, file_name)
assert os.path.exists(file_path), "Set the path and file_name variables"

nc_all = pd.read_csv(file_path, skiprows=1, sep='\t')
keep_columns = [4, 15, 26, 27] #These are hard coded sorry, see layout_ncvoter.txt
nc_all = nc_all.iloc[:, keep_columns]
nc_all.columns = [const.name, const.zipcode, const.race, const.hisp]
nc = nc_all.dropna()
nc.loc[:,const.zipcode] = nc[const.zipcode].astype(int).astype(str)
surnames = pd.Series(nc[const.name])
zips = pd.Series(nc[const.zipcode])



"""
Here are the race codes from the north carolina voter 
dataset file layout 

see layout_ncvoter.txt 
-----
A                  ASIAN
B                  BLACK or AFRICAN AMERICAN
I                  AMERICAN INDIAN or ALASKA NATIVE
M                  TWO or MORE RACES 
O                  OTHER
P                  NATIVE HAWAIIAN or PACIFIC ISLANDER
U                  UNDESIGNATED
W                  WHITE
"""
nc.loc[:,const.race] = nc[const.race].replace({
    "W": const.white,
    "B": const.black,
    "U": np.nan, #U is for undesignated
    "A": const.api,
    "P": const.api, #P is for pacific island
    "I": const.aian,
    "M": const.multiple,
    "O": np.nan
})

nc = nc.dropna(axis=0, inplace=False)

"""
Set the race value to hispanic when ethnicity is hispanic 

HL                 HISPANIC or LATINO
NL                 NOT HISPANIC or NOT LATINO
UN                 UNDESIGNATED
"""
nc.loc[nc[const.hisp] == "HL", const.race] = const.hisp

nc = nc.drop(labels=const.hisp, axis=1)
save_path = "../data/clean/nc"
save_name = "ncvoter1.csv"
nc.to_csv(os.path.join(save_path, save_name), index=False)