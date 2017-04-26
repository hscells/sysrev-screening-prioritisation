import numpy as np

from features.feature import AbstractFeature


class IDFSum(AbstractFeature):
    def idf_weights(self) -> np.ndarray:
        weights = []
        for term in self.query_vocabulary:
            if term in self.term_statistics:
                weights.append(
                    self.field_statistics['doc_count'] / self.term_statistics[term]['ttf'])
        if len(weights) == 0:
            weights = [0.0]
        return np.array(weights)

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
        return np.max(self.idf_weights())

    def feature_manager_id(self):
        return 4


class IDFMean(IDFSum):
    def calc(self) -> float:
        return np.mean(self.idf_weights())

    def feature_manager_id(self):
        return 5
