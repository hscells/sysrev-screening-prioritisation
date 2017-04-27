"""
Harry Scells
Apr 2017
"""

from features.feature import AbstractFeature


class FieldLength(AbstractFeature):
    """
    For the given field, what is the length in characters?
    """

    def calc(self) -> float:
        return len(self.field)
