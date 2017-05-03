"""
Harry Scells
Apr 2017
"""

import json
import re

from features.feature import AbstractFeature


# class QueryKeywordLength(AbstractFeature):
#     """
#     How many keywords are used in the query?
#     """
#
#     def calc(self) -> float:
#         return len(self.query_vocabulary.values())
#
#
# class QueryOperatorCount(AbstractFeature):
#     """
#     How many boolean operators are used in this query?
#     """
#
#     def get_operator_count(self, operator_name: str) -> int:
#         return len([m for m in re.finditer(operator_name, json.dumps(self.query))])
#
#     def calc(self) -> float:
#         return self.get_operator_count('"must"') + \
#                self.get_operator_count('"must_not"') + \
#                self.get_operator_count('"should"')
