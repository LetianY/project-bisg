import pandas as pd

from typing import Protocol
from abc import ABC, abstractmethod
from surgeo import SurgeoModel, BIFSGModel


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
        TARGET_SURNAME_COLUMNS = ['white', 'black', 'api', 'native', 'multiple', 'hispanic']
        df = data.copy()
        prob_df = self.model.get_probabilities(names=df['surname'], geo_df=df['ztacs'])
        df[TARGET_SURNAME_COLUMNS] = prob_df[TARGET_SURNAME_COLUMNS]
        return df
    
class WeightEstimator(ProxyPredictor):
    def __init__(self):
        self.model = None

    def inference(self, data: pd.DataFrame) -> pd.DataFrame:
        prob_df = None
        return prob_df