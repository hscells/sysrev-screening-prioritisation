import numpy as np
from typing import List

from features.feature import AbstractFeature


class IDFSum(AbstractFeature):
    def idf_weights(self) -> List[float]:
        for term in self.query_vocabulary:
            if term in self.term_statistics:
                yield self.field_statistics['doc_count'] / self.term_statistics[term]['ttf']

    def calc(self) -> float:
        return sum(self.idf_weights())

    def feature_manager_id(self) -> int:
        return 1


class IDFStd(IDFSum):
    def calc(self) -> float:
        return np.std(self.idf_weights())

    def feature_manager_id(self):
        return 2


class IDFMax(IDFSum):
    def calc(self) -> float:
        return max(self.idf_weights())

    def feature_manager_id(self):
        return 4


class IDFMean(IDFSum):
    def calc(self) -> float:
        return np.mean(self.idf_weights())

    def feature_manager_id(self):
        return 5
