"""
Harry Scells
Apr 2017
"""
import math
import numpy as np

from features.feature import AbstractFeature


class IDFSum(AbstractFeature):
    """
    Sum of the IDF scores.
    """

    def idf_weights(self) -> np.ndarray:
        """
        Compute a list of weights for a document and a query such that each weight is the sum
        of the idf for a term that appears in both a query and a document.
        :return: 
        """
        # For each term, sum the idf components for each field
        weights = {}
        fields = self.statistics['term_vectors'].keys()
        for field in fields:

            # We can now extract the statistics from the term vector API
            field_stats = self.statistics['term_vectors'][field]['field_statistics']
            term_stats = self.statistics['term_vectors'][field]['terms']

            # Normalise the fields so no key errors in the query vocabulary dict
            if field in ['population.stemmed', 'intervention.stemmed', 'outcomes.stemmed']:
                field = field.replace('.stemmed', '')
            for term in self.query_vocabulary[field]:
                if term in term_stats:
                    if term not in weights:
                        weights[term] = 0
                    weights[term] += math.log10(field_stats['doc_count'] / term_stats[term]['ttf'])

        # Next, convert it to a list
        weights = list(weights.values())
        if len(weights) == 0:
            # print('no query terms found in document')
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
