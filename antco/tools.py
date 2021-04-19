# Module containing frequently used tools already implemented.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from .base import ScoreScaler


class MinMaxScaler(ScoreScaler):
    """
    Class that performs Min-Max scaling and scales the values to the range defined by
    [min_val, max_val].

    Parameters
    ----------
    min_val: float, default=0.0
        Minimum value to scale values to.

    max_val: float, default=1.0
        Maximum value to scale values to.

    max_historic: bool, default=False
        Parameter indicating whether to use the best historical score as the maximum value for
        Min-Max scaling.
    """
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, max_historic: bool = False):
        self._min_val = min_val
        self._max_val = max_val
        self._max_historic = max_historic

    def __repr__(self):
        return f'MinMaxScaler(min_val={self._min_val}, max_val={self._max_val}, ' \
               f'max_historic={self._max_historic})'

    def scale(self, ant_scores: np.ndarray, best_historic: float):
        min_score = np.min(ant_scores)
        max_score = best_historic if self._max_historic else np.max(ant_scores)

        if min_score == max_score:  # To avoid zero division error
            return np.ones(shape=ant_scores.shape)

        return ((ant_scores - min_score) / (max_score - min_score)) * \
               (self._max_val - self._min_val) + self._min_val
