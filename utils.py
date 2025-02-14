import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from zip_codes import df_zip_crosswalk
from surgeo.models.base_model import BaseModel


# configure constants
RACE_MAPPING = {
'A': 'api',        # Asian
'B': 'black',      # Black or African American
'I': 'native',     # American Indian or Alaska Native
'M': 'multiple',   # Two or More Races
'O': 'other',      # Other - not mappable
'P': 'api',        # Native Hawaiian or Pacific Islander
'U': 'unknown',    # Undesignated - not mappable
'W': 'white'       # White
}

def load_voter_data_txt(file_path: str, delimiter: str = '\t', **kwargs) -> pd.DataFrame:
    """Load a .txt file and return a pandas DataFrame."""
    return pd.read_csv(file_path, delimiter=delimiter, **kwargs)

def clean_voter_data(df: pd.DataFrame, county_name: str='') -> pd.DataFrame:
    """Clean the voter data."""
    # filter by county_name
    print("filtering by county_name...")
    if county_name in df['county_desc'].unique():
        county_df = df[df['county_desc'] == county_name]
    else:
        county_df = df.copy()

    # all variables here: https://s3.amazonaws.com/dl.ncsbe.gov/data/layout_ncvoter.txt
    # we will only be using the following variables
    print("selecting columns...")
    usecols = ['county_id', 'county_desc', 'voter_reg_num', 'last_name', 'zip_code',
               'race_code', 'ethnic_code', 'party_cd', 'vtd_abbrv', 'vtd_desc']
    county_df = county_df[usecols]

    # clean surname
    print("cleaning surname...")
    basemodel = BaseModel()
    county_df['surname'] = county_df['last_name'].str.upper()
    county_df['surname'] = basemodel._normalize_names(county_df['surname'])

    # clean ztac
    print("cleaning ztac...")
    county_df = df_zip_crosswalk(
        dataframe=county_df, 
        zip_field_name='zip_code', 
        year=2020,
        zcta_field_name='ztacs',
        use_postalcode_if_error=False,
        suppress_prints=False
        )
    county_df['ztacs'] = county_df['ztacs'].str.zfill(5)

    # clean true race: map to Surgeo categories
    print("cleaning race...")
    county_df['true_race'] = county_df['race_code'].map(RACE_MAPPING)

    # selct and rename columns
    print("selecting and renaming columns...")
    sel_cols = ['county_id', 'county_desc', 'voter_reg_num', 'surname', 'zip_code', 'ztacs', 'race_code', 'true_race', 'party_cd']
    county_df = county_df[sel_cols]

    # remove invalid records
    print("removing invalid records...")
    return county_df.dropna(subset=['surname', 'ztacs', 'party_cd', 'true_race'])

def plot_voter_data(df: pd.DataFrame, column: str, **kwargs) -> None:
    """Result evaluations."""
    plt.hist(df[column], **kwargs)
    plt.show()





