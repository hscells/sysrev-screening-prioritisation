from features.feature import AbstractFeature


class PopulationCount(AbstractFeature):
    def terms(self, field: str) -> dict:
        if field in self.statistics['term_vectors']:
            return self.statistics['term_vectors'][field]
        else:
            return {}

    def count(self, field: str) -> float:
        i = 0.0
        doc_terms = self.terms(field)
        if len(doc_terms) == 0:
            return i
        for term in self.query_vocabulary:
            if term in doc_terms:
                i += 1
        return i

    def calc(self) -> float:
        return self.count('population')


class InterventionCount(PopulationCount):
    def calc(self) -> float:
        return self.count('intervention')


class OutcomesCount(PopulationCount):
    def calc(self) -> float:
        return self.count('outcomes')
