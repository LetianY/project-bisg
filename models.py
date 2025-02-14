import pandas as pd

from typing import Protocol
from abc import ABC, abstractmethod
from surgeo import SurgeoModel, BIFSGModel


RACE_COLS = ['white', 'black', 'api', 'native', 'multiple', 'hispanic']

class ProxyPredictor(ABC):
    @abstractmethod
    def __init__(self):
        self.model = None

    @abstractmethod
    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class BisgPredictor(ProxyPredictor):
    def __init__(self):
        self.model = SurgeoModel()

    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        prob_df = self.model.get_probabilities(names=df['surname'], geo_df=df['ztacs'])
        print(prob_df.head())
        
        sum_prob = prob_df[RACE_COLS].sum(axis=1)
        for race in RACE_COLS:
            prob_df[race] = prob_df[race] / sum_prob

        df[RACE_COLS] = prob_df[RACE_COLS]
        df['pred_race'] = df[RACE_COLS].idxmax(axis=1).fillna("unknown")
        return df