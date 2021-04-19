# Module defining the necessary interfaces related to the optimisation process.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
from abc import ABCMeta, abstractmethod
from .ant import Ant


class ObjectiveFunction(object):
    """
    Object defining the cost function to be maximised by the ant optimisation algorithm.

    Methods
    -------
    getVisitedNodes(ant: antco.Ant) -> list
        Method which, given an ant, returns a list of nodes visited by that ant.

    Attributes
    ----------
    accessory_node: bool
        Returns True if an accessor node has been added in the pre-processing of an antco.ACO
        instance, otherwise return False.

    Abstract methods
    ----------------
    evaluate(ant: antco.Ant)
        Abstract method specific to the optimisation problem to be implemented by the user. This
        method must receive an antco.Ant instance and return a scalar value. Internal modifications
        of the instance received by the function will have no effect.
    """
    __metaclass__ = ABCMeta

    _accessory_node = False

    def __repr__(self):
        return f'ObjectiveFunction'

    def __str__(self):
        return self.__repr__()

    def __call__(self, ant_idx: tuple) -> tuple:
        return self.evaluate(ant_idx[0]), ant_idx[1]

    def getVisitedNodes(self, ant: Ant):
        if self._accessory_node:
            return ant.visited_nodes[1:]
        else:
            return ant.visited_nodes

    @property
    def accessory_node(self):
        return self._accessory_node

    @accessory_node.setter
    def accessory_node(self, present: bool):
        self._accessory_node = present

    @abstractmethod
    def evaluate(self, ant: Ant) -> float:
        """
        Method that evaluates the path travelled by a given ant in such a way that it returns a
        numerical value which will be maximised.

        Parameters
        ----------
        ant: Ant
            Ant encoding the path traversed by the graph.

        Returns
        -------
        :float
            Score associated with the ant.

        Notes
        ------
        Since the framework is optimised for parallel execution, modifications to the internal
        state of the Ant during the execution of ObjectiveFunction instances will have no effect
        on the internal state.
        """
        raise NotImplementedError


