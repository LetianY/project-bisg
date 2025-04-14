import os
import re
import pandas as pd
from src.constants import constants as const

make_counts = False
path = "../data/raw/census"
file_name = "zipcode_race_raw.csv"
file_path = os.path.join(path, file_name)
assert os.path.exists(file_path), "Set the path and file_name variables"

df = pd.read_csv(file_path)

#Hard code, column values in "DECENNIALSF12010.P8-Column-Metadata.csv"
"""
NAME	- Geographic Area Name
P009001	- Total
P009002	- Total!!Hispanic or Latino
P009005 - Total!!Not Hispanic or Latino!!Population of one race!!White alone
P009006 - Total!!Not Hispanic or Latino!!Population of one race!!Black or African American alone
P009007	- Total!!Not Hispanic or Latino!!Population of one race!!American Indian and Alaska Native alone
P009008	- Total!!Not Hispanic or Latino!!Population of one race!!Asian alone
P009009	- Total!!Not Hispanic or Latino!!Population of one race!!Native Hawaiian and Other Pacific Islander alone
P009011	- Total!!Not Hispanic or Latino!!Two or More Races
"""
keep_columns = ["NAME", "P009001", "P009002", "P009005", "P009006", "P009007", "P009008", "P009009", "P009011"]

df = df[keep_columns].iloc[1:]

#Rename columns
df = df.rename(columns={
    "NAME": const.zipcode,
    "P009001": const.count,
    "P009002": const.hisp,
    "P009005": const.white,
    "P009006": const.black,
    "P009007": const.aian,
    "P009008": const.api,
    "P009009": const.nhpi,
    "P009011": const.multiple
})

df[const.zipcode] = df[const.zipcode].map(lambda x: x.split(" ")[1])

#Set everything but zipcode to int
df.iloc[:,1:] = df.iloc[:,1:].astype(int)

#reset index
df.reset_index()

#Remove rows that do not have a count
df = df[df[const.count] != 0]


#update asian prob to include aian
df[const.api] += df[const.nhpi]
df = df.drop(columns=const.nhpi)
df[const.zipcode] = df[const.zipcode].astype(str)


"""
Datafarame contains the counts of each individual race 
Convert to PR[G|R] by dividing by the total number of individuals 
belonging to each racial category 
"""
save_name = "zipcodes.csv"
save_path = "../data/clean/census"
if not make_counts:
    df.iloc[:,2:] = df.iloc[:,2:].div(df.iloc[:,2:].sum(axis=0), axis=1)
else:
    save_name = "zipcodes_counts.csv"


#ReOrder Columns
df.iloc[:,2:] = df[const.races]
df.columns = df.columns[:2].to_list() + const.races


df.to_csv(os.path.join(save_path, save_name), index=False)