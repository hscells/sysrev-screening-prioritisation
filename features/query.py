"""
Harry Scells
Apr 2017
"""

import json
import re
from collections import OrderedDict

from features.feature import AbstractFeature
from ltrfeatures import generate_query_vocabulary


class QueryKeywordLength(AbstractFeature):
    """
    How many keywords are used in the query?
    """

    def calc(self) -> float:
        vocab = generate_query_vocabulary([OrderedDict([('query', self.query)])])
        return len(vocab[self.field])


class QueryOperatorCount(AbstractFeature):
    """
    How many boolean operators are used in this query?
    """

    def get_operator_count(self, operator_name: str) -> int:
        return len([m for m in re.finditer(operator_name, json.dumps(self.query))])

    def calc(self) -> float:
        return self.get_operator_count('"must"') + \
               self.get_operator_count('"must_not"') + \
               self.get_operator_count('"should"')
