from features.feature import AbstractFeature


class PopulationCount(AbstractFeature):
    def terms(self, field: str) -> dict:
        """
        Get a list of terms for a field in a document.
        :param field: 
        :return: 
        """
        if field in self.statistics['term_vectors']:
            return self.statistics['term_vectors'][field]
        else:
            return {}

    def count(self, field: str) -> float:
        """
        Count the number of query terms for a given PICO field are in the same PICO field
        as the document.
        :param field: 
        :return: 
        """
        i = 0.0
        doc_terms = self.terms(field)
        if len(doc_terms) == 0:
            return i
        # Normalise the fields so no key errors in the query vocabulary dict
        if field in ['population.stemmed', 'intervention.stemmed', 'outcomes.stemmed']:
            field = field.replace('.stemmed', '')
        for term in self.query_vocabulary[field]:
            if term in doc_terms['terms'].keys():
                i += 1
        return i

    def calc(self) -> float:
        return self.count('population.stemmed')


class InterventionCount(PopulationCount):
    def calc(self) -> float:
        return self.count('intervention.stemmed')


class OutcomesCount(PopulationCount):
    def calc(self) -> float:
        return self.count('outcomes.stemmed')
