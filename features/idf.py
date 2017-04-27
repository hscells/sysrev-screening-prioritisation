"""
Harry Scells
Apr 2017
"""
import numpy as np

from features.feature import AbstractFeature


class IDFSum(AbstractFeature):
    """
    Sum of the IDF scores.
    """

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


class IDFStd(IDFSum):
    """
    Std. dev of the IDF scores.
    """

    def calc(self) -> float:
        return np.std(self.idf_weights())


class IDFMax(IDFSum):
    """
    Max of the IDF scores.
    """

    def calc(self) -> float:
        return np.max(self.idf_weights())


class IDFMean(IDFSum):
    """
    Mean of the IDF scores.
    """

    def calc(self) -> float:
        return np.mean(self.idf_weights())
