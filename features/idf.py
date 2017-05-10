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
        # For each term, sum the idf components for each field
        weights = {}
        fields = self.statistics['term_vectors'].keys()
        for field in fields:
            field_stats = self.statistics['term_vectors'][field]['field_statistics']
            term_stats = self.statistics['term_vectors'][field]['terms']
            for term in self.query_vocabulary:
                if term in term_stats:
                    if term not in weights:
                        weights[term] = 0
                    weights[term] += field_stats['doc_count'] / term_stats[term]['ttf']

        # Next, convert it to a list
        weights = list(weights.values())
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
