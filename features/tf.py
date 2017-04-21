from features.feature import AbstractTermFeature


class TermFrequencyFeature(AbstractTermFeature):
    def calc(self) -> float:
        """
        |D| = \sum_{t\in D}tf(t)
        \alpha tf(t) = \frac{tf(t)}{|D|}
            
        Since we can't use plain TF, we need to normalise somehow. To do this, we calculate the
        document length and divide: TF(t)/DL
        """
        if self.term in self.term_statistics:
            dl = sum([self.term_statistics[term]['term_freq'] for term in self.term_statistics])
            normalised_tf = self.term_statistics[self.term]['term_freq'] / dl
            return normalised_tf
        return 0.0
