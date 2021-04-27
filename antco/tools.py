# Module containing frequently used tools already implemented.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import matplotlib.pyplot as plt
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


class HyperparameterCheckerAS(object):
    """
    Class to plot the effects of the hyperparameters associated with the Ant System (AS) pheromone
    updating strategy on pheromone updating.

    The plot() method will plot two graphs, the first one representing the increase in the values
    of the pheromone matrix assuming that all ants used for its update are used, i.e. in a real
    case all ants that have visited the same connection, and the second one the decrease in the
    value of the pheromone matrix when no ant visits a given connection during all the plotted
    interactions.
    In the increment graph, several lines will be plotted, each corresponding to the random
    assignment of a score to each ant in the range of values indicated in the legend.

    Parameters
    ----------
    n_ants: int
        Number of ants used for updating the pheromone matrix.

    pher_init: float
        Initial value of the pheromone matrix.

    evaporation: float
        Pheromone evaporation parameter.

    weight: float, default=1.0
        Weight used to model the contribution of each ant to the pheromone update (it is assumed
        that the ants' scores have been scaled to the range [0,1]).

    seed: int, default=None
        Random seed.

    Methods
    -------
    plot(num_iterations: int = 100, figsize: tuple = (10, 10), linewidth: float = 2.5,
        title_size: int = 15, save_plot: str = None):

        Method to plotting the increase and decrease of pheromone matrix values over specified
        interactions.
    """
    def __init__(self, n_ants: int, pher_init: float, evaporation: float, weight: float = 1.0,
                 seed: int = None):
        self.n_ants = n_ants
        self.pher_init = pher_init
        self.evaporation = evaporation
        self.weight = weight
        self.seed = seed

    def plot(self, num_iterations: int = 100, figsize: tuple = (10, 10), linewidth: float = 2.5,
             title_size: int = 15, save_plot: str = None):
        """
        Parameters
        ----------
        num_iterations: int, default=100
            Number of interactions to simulate.

        figsize: tuple, default=(10, 10)
            Tuple indicating the size of the figure.

        linewidth: float, default=2.5
            Thickness of the lines in the plot.

        title_size: int, default=15
            Size of the title of each graphic.

        save_plot: str, default=None
            File in which to save the generated graph, if no value is provided the graph will not
            be saved.
        """
        iterations = [it for it in range(num_iterations)]
        pher_increase_100 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            weight=self.weight, iterations=iterations, score_range=(0.99, 1.0), seed=self.seed)
        pher_increase_75 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            weight=self.weight, iterations=iterations, score_range=(0.75, 1.0), seed=self.seed)
        pher_increase_50 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            weight=self.weight, iterations=iterations, score_range=(0.50, 1.0), seed=self.seed)
        pher_increase_25 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            weight=self.weight, iterations=iterations, score_range=(0.25, 1.0), seed=self.seed)
        pher_decrease = self._pherDecrease(
            pher_init=self.pher_init, evaporation=self.evaporation, iterations=iterations)

        fig, axes = plt.subplots(2, figsize=figsize)
        axes[0].plot(iterations, pher_increase_100, label='Scores [0.99-1.0]', color='#641E16',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_75, label='Scores [0.75-1.0]', color='#A93226',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_50, label='Scores [0.50-1.0]', color='#D98880',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_25, label='Scores [0.25-1.0]', color='#F2D7D5',
                     linewidth=linewidth)
        axes[1].plot(iterations, pher_decrease, color='#2980B9', linewidth=linewidth)

        axes[0].set_title('Pheromone increase', fontsize=title_size)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend()
        axes[1].set_title('Pheromone decrease', fontsize=title_size)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        if save_plot is not None:
            plt.savefig(save_plot, dpi=150)

        plt.show()

    @staticmethod
    def _pherIncrease(n_ants: int, pher_init: float, evaporation: float, weight: float,
                      iterations: list, score_range: tuple, seed: int):
        """ Simulate the pheromone update (increasing the values) according to the AS update
        strategy. """
        if seed is not None:
            np.random.seed(seed)
        pher_increase = []
        pher = pher_init
        for it in iterations:
            score = np.sum(
                np.random.uniform(low=score_range[0], high=score_range[1], size=n_ants) * weight)
            pher = (1 - evaporation) * pher + score * weight
            pher_increase.append(pher)

        return pher_increase

    @staticmethod
    def _pherDecrease(pher_init: float, evaporation: float, iterations: list):
        """ Simulate the pheromone update (decreasing the values) according to the AS update
        strategy. """
        pher_decrease = []
        pher = pher_init
        for it in iterations:
            pher = (1 - evaporation) * pher
            pher_decrease.append(pher)

        return pher_decrease


class HyperparameterCheckerMMAS(object):
    """
    Class to plot the effects of the hyperparameters associated with the Min-Max Ant System (MMAS)
    pheromone updating strategy on pheromone updating.

    The plot() method will plot two graphs, the first one representing the increase in the values
    of the pheromone matrix assuming that all ants used for its update are used, i.e. in a real
    case all ants that have visited the same connection, and the second one the decrease in the
    value of the pheromone matrix when no ant visits a given connection during all the plotted
    interactions.
    In the increment graph, several lines will be plotted, each corresponding to the random
    assignment of a score to each ant in the range of values indicated in the legend.

    Parameters
    ----------
    n_ants: int
        Number of ants used for updating the pheromone matrix.

    pher_init: float
        Initial value of the pheromone matrix.

    evaporation: float
        Pheromone evaporation parameter.

    limits: tuple
        Bounds of the MMAS update strategy.

    weight: float, default=1.0
        Weight used to model the contribution of each ant to the pheromone update (it is assumed
        that the ants' scores have been scaled to the range [0,1]).

    seed: int, default=None
        Random seed.

    Methods
    -------
    plot(num_iterations: int = 100, figsize: tuple = (10, 10), linewidth: float = 2.5,
        title_size: int = 15, save_plot: str = None):

        Method to plotting the increase and decrease of pheromone matrix values over specified
        interactions.
    """
    def __init__(self, n_ants: int, pher_init: float, evaporation: float, limits: tuple,
                 weight: float = 1.0, seed: int = None):
        self.n_ants = n_ants
        self.pher_init = pher_init
        self.evaporation = evaporation
        self.limits = limits
        self.weight = weight
        self.seed = seed

    def plot(self, num_iterations: int = 100, figsize: tuple = (10, 10), linewidth: float = 2.5,
             title_size: int = 15, save_plot: str = None):
        """
        Parameters
        ----------
        num_iterations: int, default=100
            Number of interactions to simulate.

        figsize: tuple, default=(10, 10)
            Tuple indicating the size of the figure.

        linewidth: float, default=2.5
            Thickness of the lines in the plot.

        title_size: int, default=15
            Size of the title of each graphic.

        save_plot: str, default=None
            File in which to save the generated graph, if no value is provided the graph will not
            be saved.
        """
        iterations = [it for it in range(num_iterations)]
        pher_increase_100 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            limits=self.limits, weight=self.weight, iterations=iterations, score_range=(0.99, 1.0),
            seed=self.seed)
        pher_increase_75 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            limits=self.limits, weight=self.weight, iterations=iterations, score_range=(0.75, 1.0),
            seed=self.seed)
        pher_increase_50 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            limits=self.limits, weight=self.weight, iterations=iterations, score_range=(0.50, 1.0),
            seed=self.seed)
        pher_increase_25 = self._pherIncrease(
            n_ants=self.n_ants, pher_init=self.pher_init, evaporation=self.evaporation,
            limits=self.limits, weight=self.weight, iterations=iterations, score_range=(0.25, 1.0),
            seed=self.seed)
        pher_decrease = self._pherDecrease(
            pher_init=self.pher_init, evaporation=self.evaporation, limits=self.limits,
            iterations=iterations)

        fig, axes = plt.subplots(2, figsize=figsize)
        axes[0].plot(iterations, pher_increase_100, label='Scores [0.99-1.0]', color='#641E16',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_75, label='Scores [0.75-1.0]', color='#A93226',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_50, label='Scores [0.50-1.0]', color='#D98880',
                     linewidth=linewidth)
        axes[0].plot(iterations, pher_increase_25, label='Scores [0.25-1.0]', color='#F2D7D5',
                     linewidth=linewidth)
        axes[1].plot(iterations, pher_decrease, color='#2980B9', linewidth=linewidth)

        axes[0].set_title('Pheromone increase', fontsize=title_size)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].legend()
        axes[1].set_title('Pheromone decrease', fontsize=title_size)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        if save_plot is not None:
            plt.savefig(save_plot, dpi=150)

        plt.show()

    @staticmethod
    def _pherIncrease(n_ants: int, pher_init: float, evaporation: float, limits: tuple, weight: float,
                      iterations: list, score_range: tuple, seed: int):
        """ Simulate the pheromone update (increasing the values) according to the MMAS update
        strategy. """
        if seed is not None:
            np.random.seed(seed)
        pher_increase = []
        pher = pher_init
        for it in iterations:
            scores = np.random.uniform(low=score_range[0], high=score_range[1], size=n_ants)
            for score in scores:
                pher = (1 - evaporation) * pher + score * weight
            if pher < limits[0]:
                pher = limits[0]
            if pher > limits[1]:
                pher = limits[1]
            pher_increase.append(pher)

        return pher_increase

    @staticmethod
    def _pherDecrease(pher_init: float, evaporation: float, limits: tuple, iterations: list):
        """ Simulate the pheromone update (decreasing the values) according to the MMAS update
        strategy. """
        pher_decrease = []
        pher = pher_init
        for it in iterations:
            pher = (1 - evaporation) * pher
            if pher < limits[0]:
                pher = limits[0]
            if pher > limits[1]:
                pher = limits[1]
            pher_decrease.append(pher)

        return pher_decrease
