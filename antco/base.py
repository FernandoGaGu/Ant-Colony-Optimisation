# Module that collects the interfaces that must follow certain components used in the framework.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from abc import ABCMeta, abstractmethod


class DecaySchedule(object):
    """
    Class representing a decay schedule that is applied to the score values or evaporation
    parameter which are used for update the pheromone values in each iteration.

    Methods
    -------
    show()
        Method that allows to represent how the weighting evolves as the iterations progress
        according to the definition given by the user of the abstract decay method.

    Abstract methods
    ----------------
    decay(iteration: int) -> float

    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return f'DecaySchedule'

    def __str__(self):
        return self.__repr__()

    def __call__(self, iteration: int) -> float:
        return self.decay(iteration)

    @abstractmethod
    def decay(self, iteration: int) -> float:
        """
        Method that must return a numeric value that will act as a decay on the scores or
        evaporation parameter. In this way the scores or evaporation will be multiplied by the
        value received. This allow scores or evaporation to be smoothed as a function of the
        current iteration. Depending on which iteration the algorithm is in the pheromones
        will be updated in a more abrupt or more smooth way depending on the value returned.

        Parameters
        ----------
        iteration: int
            Algorithm iteration.

        Returns
        -------
        :float
            Decay value
        """
        raise NotImplementedError

    def show(self, from_it: int = 1, to_it: int = 100, title: str = 'Decay schedule'):
        """
        Method that shows a plot of how the weighting applied by the decay evolves as a function
        of the number of interactions.

        Parameters
        ----------
        from_it: int, default=1
            Initial iteration.

        to_it: int, default=100
            Last iteration.

        title: str, default='Decay schedule'
            Plot title.
        """
        assert from_it > 0, 'Initial iteration (from_it parameter) must be > 1.'
        assert from_it < to_it, 'Initial iteration (from_it parameter) must be > final iteration (to_it parameter).'

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(from_it, to_it)
        y = [self.decay(it) for it in x]

        ax.scatter(x, y, c=plt.cm.Oranges(y))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(title)

        plt.show()


class ScoreScaler(object):
    """
    Class defining the interface followed by those classes used to scale the values of the scores
    given by the cost function to a given range.

    Abstract methods
    ----------------
    scale(ant_scores: np.ndarray, best_historic: float)
    """
    __metaclass__ = ABCMeta

    def __repr__(self):
        return f'ScoreScaler'

    def __str__(self):
        return self.__repr__()

    def __call__(self, ant_scores: np.ndarray, best_historic: float):
        return self.scale(ant_scores, best_historic)

    def scale(self, ant_scores: np.ndarray, best_historic: float):
        """
        Method that receives as arguments an array with the scores given by the cost function and
        the best score seen by the algorithm and must return an array of the same size as the one
        received with the scores scaled as decided by the user.

        Parameters
        ----------
        ant_scores: np.ndarray (n_ants), dtype=np.float64
            Scores given by the objective function.

        best_historic: float
            Best score found until the current iteration.

        Returns
        -------
        :np.ndarray (n_ants), dtype=np.float64
            Scaled scores.
        """
        raise NotImplementedError
