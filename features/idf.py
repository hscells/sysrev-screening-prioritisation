from features.feature import AbstractFeature


class IDFSum(AbstractFeature):
    """
    
    """

    def calc(self) -> float:
        total = 0.0
        for term in self.query_vocabulary:
            if term in self.term_statistics:
                total += self.field_statistics['doc_count'] / self.term_statistics[term]['ttf']
        return total

    def feature_manager_id(self) -> int:
        return 1
