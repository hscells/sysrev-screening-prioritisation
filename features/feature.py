"""
These abstract base classes are here to extend with other classes in this features folder. The
`ltrfeatures.py` script will use this module to extract features from elasticsearch. These
abstract classes are ignored by the script.

For an example, have a look a the `tf.py` script in this folder.

Harry Scells
Apr 2017
"""


# TODO: Automatically find these abstract classes instead of hardcoding them.

class AbstractFeature(object):
    """
    This is the abstract class for performing feature engineering. The calc function is called
    during the feature selection stage. Classes that extend this AbstractFeature class that are
    inside the `features` folder will be automatically loaded and calculated as part of the
    `ltrfeatures.py` script.
    """

    def calc(self) -> float:
        raise NotImplementedError()


class AbstractTermFeature(AbstractFeature):
    """
    Extract features with respect to terms and the term vector api of elasticsearch.
    
    For information on what the statistics object looks like, see:
    https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-termvectors.html#_example_returning_stored_term_vectors
    """

    def __init__(self, statistics: dict, doc: str, field: str, term: str):
        self.statistics = statistics
        self.doc = doc
        self.field = field
        self.term = term

        self.field_statistics = statistics['term_vectors'][field]['field_statistics']
        self.term_statistics = statistics['term_vectors'][field]['terms']

    def calc(self) -> float:
        raise NotImplementedError()


class AbstractQueryFeature(AbstractFeature):
    """
    Extract features with respect to the query.
    """

    def __init__(self, query: dict):
        self.query = query

    def calc(self) -> float:
        raise NotImplementedError()
