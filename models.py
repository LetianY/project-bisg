import os
import sys
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from const import constants
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from typing import Protocol
from abc import ABC, abstractmethod
from surgeo import SurgeoModel, BIFSGModel
from references.proxies.src.proxies import ftBisg

# use WRU race columns
RACE_COLS = constants.RACE_COLS_WRU

class ProxyPredictor(ABC):
    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def normalize_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        sum_prob = data[RACE_COLS].sum(axis=1)
        for race in RACE_COLS:
            if race in data.columns:
                data[race] = data[race] / sum_prob
        return data

    def generate_race_label(self, data: pd.DataFrame) -> pd.DataFrame:
        data['pred_race'] = data[RACE_COLS].idxmax(axis=1).fillna("unknown")
        return data

class SurgeoPredictor(ProxyPredictor):
    """
    This class is a wrapper around the SurgeoModel and provides a method for inference on a given DataFrame.
    """
    def __init__(self):
        self.model = SurgeoModel()

    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        prob_df = self.model.get_probabilities(names=df['surname'], geo_df=df['ztacs'])
        print(prob_df.head())
        
        # sum_prob = prob_df[RACE_COLS].sum(axis=1)
        # for race in RACE_COLS:
        #     prob_df[race] = prob_df[race] / sum_prob

        for race in RACE_COLS:
            if race in prob_df.columns:
                df[race] = prob_df[race]
            else:
                df[race] = 0.0

        df[RACE_COLS] = prob_df[RACE_COLS]
        df['pred_race'] = df[RACE_COLS].idxmax(axis=1).fillna("unknown")
        return df

class WRUPredictor(ProxyPredictor):
    """
    This class is a wrapper around the WRU package and provides a method for race inference.
    """
    def __init__(self, census_api_key=None):
        """
        Initialize WRU Model with R backend.
        
        Args:
            census_api_key (str, optional): Census API key for accessing census data.
                If not provided, will check for CENSUS_API_KEY environment variable.
        """
        # Initialize R environment
        pandas2ri.activate()
        
        # Set Census API key
        self.census_api_key = census_api_key or os.environ.get('CENSUS_API_KEY')
        if not self.census_api_key:
            raise ValueError("Census API key must be provided either as argument or as CENSUS_API_KEY environment variable")
        
        # Import R packages
        self.wru = importr('wru')
        self.dplyr = importr('dplyr')
        self.future = importr('future')
        
        # Set up parallel processing in R
        self.future.plan(self.future.multisession)
        
        # Set census API key in R environment
        ro.r(f'Sys.setenv(CENSUS_API_KEY = "{self.census_api_key}")')

        # Initialize census data
        self.census_data = None

    def _prepare_data(self, data: pd.DataFrame) -> ro.vectors.DataFrame:
        """
        Prepare the input dataframe for WRU processing.
        
        Args:
            data (pd.DataFrame): Input data containing at minimum surname and zcta columns
            
        Returns:
            ro.vectors.DataFrame: R dataframe prepared for WRU processing
        """
        
        df = data.copy()
        
        # Rename columns if needed (e.g., zcta5 -> zcta)
        if 'zcta5' in df.columns and 'zcta' not in df.columns:
            df.rename(columns={'zcta5': 'zcta'}, inplace=True)

        if 'zctas' in df.columns and 'zcta' not in df.columns:
            df['zcta'] = df['zctas']

        if 'zcta' in df.columns:
            df['zcta'] = df['zcta'].str.zfill(5)
        
        # Ensure proper data types to avoid R coercion issues
        # Ensure surname is string type
        if 'surname' in df.columns:
            df['surname'] = df['surname'].str.upper()
        
        # Make sure any potential geographic identifiers are all strings
        geo_cols = ['county', 'tract', 'block', 'block_group', 'place', 'zcta']
        for col in geo_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        # Convert to R dataframe
        r_df = pandas2ri.py2rpy(df)

        # Remove df copy
        del df

        return r_df

    def get_census_data(self, state="NC", year="2020", census_geo="zcta"):
        """
        Get census data for the specified state and year.
        
        Args:
            state (str): State abbreviation (default: "NC")
            year (str): Census year (default: "2020")
            census_geo (str): Census geography type (default: "zcta")
            
        Returns:
            rpy2 object: Census data from WRU package
        """
        # Get census data using R's WRU package
        census_data = self.wru.get_census_data(
            year=year,
            key=self.census_api_key,
            states=state,
            census_geo=census_geo,
            age=False,
            sex=False,
            retry=3,
            county_list=ro.r('NULL')
        )
        return census_data

    def inference(
            self, 
            data: pd.DataFrame = None, 
            model='BISG', 
            state="NC", 
            year="2020", 
            census_geo="zcta",
            census_surname=True, 
            surname_only=False, 
            names_to_use="surname",
            impute_missing=False, 
            skip_bad_geos=True, 
            use_counties=False,
            reload_census=True
        ) -> pd.DataFrame:
        """
        Get race probabilities using WRU's predict_race function.

        Args:
            data (pd.DataFrame, optional): DataFrame with required columns. Must contain a column named 'surname' for each individual's surname.
                If using geolocation, must contain a 'state' column with two-character state abbreviations (e.g., "NC").
                Additional columns may include 'county', 'tract', 'block', 'block_group', 'place', or 'zcta' if geographic data is used.
            model (str, optional): Model to use for prediction, either "BISG" (default) or "fBISG" for error-correction and fully-Bayesian model.
            state (str, optional): State abbreviation for Census data (default: "NC").
            year (str, optional): Census year to use, either "2010" or "2020" (default: "2020").
            census_geo (str, optional): Geographic level to use for merging Census data. Options: "tract", "block", "block_group", 
                "county", "place", or "zcta" (default: "zcta").
            census_surname (bool, optional): Whether to use the U.S. Census Surname List for race prediction (default: True).
            surname_only (bool, optional): Whether to use only surname data for race prediction (default: False).
            names_to_use (str, optional): Names to use for prediction. One of "surname", "surname, first", or "surname, first, middle" (default: "surname").
            impute_missing (bool, optional): Whether to impute missing values (default: False).
            skip_bad_geos (bool, optional): Whether to skip any geolocations not present in the Census data (default: True).
            use_counties (bool, optional): Whether to filter Census data by counties available in the data (default: False).
            reload_census (bool, optional): Whether to reload census data (default: True).

        Returns:
            pd.DataFrame: DataFrame with race probabilities, including the estimated probability for each race category.
        """
        # Create dataframe if not provided
        if data is None:
            raise ValueError("Data must be provided")
        
        # df = data.copy()

        # Prepare data for WRU
        r_df = self._prepare_data(data)
        
        # Get or retrieve cached census data
        if reload_census or self.census_data is None:
            print("Getting census data...")
            self.census_data = self.get_census_data(state=state, year=year, census_geo=census_geo)
            
        # Run WRU predict_race
        print("Predicting race...")
        result = self.wru.predict_race(
                voter_file=r_df,
                census_surname=census_surname,
                surname_only=surname_only,
                census_geo=census_geo,
                year=year,
                census_key=self.census_api_key,
                census_data=self.census_data,
                model=model,
                names_to_use=names_to_use,
                age=False,
                sex=False,
                party=ro.r('NULL'),
                retry=3,
                impute_missing=impute_missing,
                skip_bad_geos=skip_bad_geos,
                use_counties=use_counties,
            )
                
        # Convert R result to pandas
        result_df = pandas2ri.rpy2py(result)
        
        # Map WRU column names to our standard format
        column_map = {
            'pred.whi': 'white',
            'pred.bla': 'black',
            'pred.his': 'hispanic',
            'pred.asi': 'api',
            'pred.oth': 'other',  # WRU has 'other' instead of 'multiple'
        }
        result_df = result_df.rename(columns=column_map)

        for race in RACE_COLS:
            if race not in result_df.columns:
                result_df[race] = 0.0

        return result_df
    
class ZipWRUextPredictor(ProxyPredictor):
    """
    A wrapper class around the WRU extention package: zipWRUext.
    """

    
class cBISGPredictor(ProxyPredictor):
    """
    This class implements the cBISG proxy estimation method.
    The estimation method is based on the following equation:
        Pr[R∣G,S,Y] ∝ Pr[R∣G,Y]⋅Pr[S∣R]
    
    where:
        - R is race
        - G is geographic information
        - S is surname
        - Y is contexual information (e.g., party affiliation)
    
    The original paper is: https://arxiv.org/pdf/2409.01984
    "Observing Context Improves Disparity Estimation when Race is Unobserved" by Kweku Kwegyir-Aggrey et al.

    This class is a wrapper around the cBISG model and provides a method for inference on a given DataFrame.
    Original repo: https://github.com/kwekuka/proxies/
    """
    def __init__(self, census_api_key):
        # self.model = ftBisg(races=None)
        pass

    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        pass