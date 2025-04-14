import os
import pandas as pd
from src.constants import constants


data_path = "../data/hmda/state_MD-PA-NY-DE-VA.csv"

assert os.path.exists(data_path), "Make sure the data exists"

hmda = pd.read_csv(data_path, dtype={
    'census_tract': str,
    'applicant_ethnicity-1':str,
    'applicant_race-1': str
})

categorial_features = [
    "derived_loan_product_type",
    "loan_type",
    "loan_purpose",
    "lien_status",
    "open-end_line_of_credit",
    "business_or_commercial_purpose",
    "hoepa_status",
    "interest_only_payment",
    "balloon_payment",
    "total_units",
    "denial_reason-1"
]

continuous_features = [
    "loan_amount",
    "income",
]

outcome = [
    "action_taken",
    "denial_reason-1"
]

demographics = [
    "state_code",
    "county_code",
    "census_tract",
    "derived_ethnicity",
    "derived_race",
    "derived_sex",
]

hmda = hmda[demographics + continuous_features + outcome]

hmda["derived_race"].value_counts()
hmda = hmda[hmda["derived_race"] != "Race Not Available"]
hmda["derived_race"] = hmda["derived_race"].replace({
    "White": constants.white,
    "Black or African American": constants.black,
    "Asian": constants.api,
    "Joint": constants.multiple,
    "Native Hawaiian or Other Pacific Islander": constants.api,
    "American Indian or Alaska Native": constants.aian,
    "2 or more minority races": constants.multiple
})

hmda = hmda[hmda["derived_race"] != "Free Form Text Only"]
mask = hmda["derived_ethnicity"] != "Hispanic or Latino"
hmda["derived_race"] = hmda["derived_race"].where(mask, constants.hisp)
hmda["derived_race"].value_counts()

hmda.rename(columns={"derived_race": "constants.race"})
hmda = hmda.drop(["derived_ethnicity"], axis=1)

hmda["tract"] = hmda["census_tract"].apply(lambda x: str(x)[5:])
hmda["state"] = hmda["census_tract"].apply(lambda x: str(x)[:2])
hmda["county"] = hmda["census_tract"].apply(lambda x: str(x)[2:5])

hmda = hmda.reset_index(drop=True)
hmda["derived_race"].value_counts()
hmda.to_csv("../data/hmda/state_MD-PA-NY-DE-VA.csv")