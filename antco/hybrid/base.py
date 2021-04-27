# Module containing the interface to follow for the implementation of metaheuristics to run hybrid
# Ant Colony based algorithms.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import joblib
import multiprocessing as mp
import numpy as np
from antco.optim import ObjectiveFunction
from abc import ABCMeta, abstractmethod


class MetaHeuristic(object):
    """
    Class defining the interface to follow to implement a meta-heuristic that can be coupled to ant
    colony optimisation (hybrid approaches).

    Parameters
    ----------
    objective_function: antco.optim.ObjectiveFunction
        Objective function.

    add_to_old: bool, default=False
        Indicates whether to add the ants given by the hybrid to the ants received by the
        ant optimisation algorithm. By default only the ants returned by the hybrid
        strategy will be returned to the ACO algorithm for pheromone update.

    Abstract methods
    -------
    optimise(ants: list) -> list
         Method that receives a list of ants (whose attributes can be accessed through the methods
         defined for the antco.Ant class), applies the hybrid optimisation strategy
         implemented by the user and must return a tuple where the first element must correspond to
         a list of ants whose solutions must have been modified through the hybrid strategy, and
         the second element must correspond to the scores (in the same order) associated with the
         ants provided in the first parameter.
    """
    __metaclass__ = ABCMeta

    def __init__(self, objective_function: ObjectiveFunction, add_to_old: bool = False):
        assert isinstance(objective_function, ObjectiveFunction), \
            'objective_function must be a subclass of antco.optim.ObjectiveFunction'

        self.evaluate = objective_function
        self.__add_to_old = add_to_old

    def __repr__(self):
        return 'MetaHeuristic'

    def __str__(self):
        return self.__repr__()

    def __call__(self, ants: list) -> tuple:
        """
        Parameters
        ----------
        ants: list
           List of antco.Ant instances.

        Returns
        -------
        :tuple
            Returns a tuple where the first element corresponds to a list of ants and the second to
            the scores associated with each ant.
        """

        # Initial evaluation of the ants
        scores = self._evaluate(ants)

        # Apply hybrid strategy
        improved_ants, improved_scores = self.optimise(ants, scores)

        if self.evaluate.accessory_node:  # If an accessory node is being used, add to all ants
            init_pos = [ants[0].initial_position]
            for ant in improved_ants:
                ant.visited_nodes = init_pos + ant.visited_nodes.tolist()

        # Merge improved ants and old ants
        if self.__add_to_old:
            return ants + improved_ants, np.append(scores, improved_scores)

        return improved_ants, improved_scores

    def _evaluate(self, ants: list):
        """ Method to assess the quality of the received ant list using the defined objectiveFunction
        function."""
        assert len(ants) > 0, 'At least one ant must be passed to the hybrid'
        scores = np.array([
            self.evaluate((ant, idx)) for idx, ant in enumerate(ants)], dtype=np.float64)[:, 0]

        return scores

    @abstractmethod
    def optimise(self, ants: list, scores: list) -> tuple:
        """
        Method that using the paths travelled by the ants must return a list with an arbitrary
        number of ants (note that the number of ants returned does not have to coincide with the
        number of ants received).

        Parameters
        ----------
        ants: list
           List of antco.Ant instances.

        scores: list
            List of scores associated with each ant.

        Returns
        -------
        :tuple
            Returns a tuple where the first element corresponds to the list of improved ants and
            the second element to the scores given by the cost function. Note that if the evaluation
            procedure of the cost function defined in the hybrid strategy does not coincide with
            the cost function defined in the ACO algorithm, strange behaviours may occur.
        """
        raise NotImplementedError
