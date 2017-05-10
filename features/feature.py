"""
This abstract base class is here to extend with other classes in this features folder. The
`ltrfeatures.py` script will use this module to extract features from elasticsearch. This
abstract class is ignored by the script.

For an example, have a look a the `idf.py` script in this folder.

Harry Scells
Apr 2017
"""


# TODO: Automatically find the abstract classes instead of hard coding it.


class AbstractFeature(object):
    """
    This is the abstract class for performing feature engineering. The calc function is called
    during the feature selection stage. Classes that extend this AbstractFeature class that are
    inside the `features` folder will be automatically loaded and calculated as part of the
    `ltrfeatures.py` script.
    """

    def __init__(self, statistics: dict, query: dict, query_vocabulary: dict):
        self.statistics = statistics
        self.query = query
        self.query_vocabulary = query_vocabulary

    def calc(self) -> float:
        raise NotImplementedError()
