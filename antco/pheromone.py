# Module that implements the different strategies for updating pheromone values.
#
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from .ant import Ant
from .c_pheromone import (
    updateUndAS,
    updateDirAS,
    updateUndMMAS,
    updateDirMMAS,
    updateUndEliteAS,
    updateDirEliteAS,
    updateUndEliteMMAS,
    updateDirEliteMMAS,
    updateUndACS,
    updateDirACS)


def updateAS(topology: str, elite: bool = False):
    """
    Function that returns the Ant System pheromone value update strategy optimised on the basis of
    the graph topology received as an argument.

    Parameters
    ----------
    topology: str
        Graph topology: 'directed' or 'D' and 'undirected' or 'U'.

    elite: bool (default False)
        Indicates whether to use only the best ants for the pheromone update.

    Returns
    -------
    :function
        Algorithm optimised to work with the type of topology specified.
    """
    assert isinstance(topology, str), 'topology argument must be a string.'

    topology_ = topology.lower()

    if topology_ == 'd' or topology_ == 'directed':
        if elite:
            return updateDirEliteAS
        return updateDirAS
    elif topology_ == 'u' or topology_ == 'undirected':
        if elite:
            return updateUndEliteAS
        return updateUndAS
    else:
        assert False, 'Unrecognised topology parameter. Available options are: "directed" or "D" ' \
                      'and "undirected" or "U".'


def updateMMAS(topology: str, elite: bool = False):
    """
    Function that returns the MIN-MAX Ant System pheromone value update strategy optimised on the
    basis of the graph topology received as an argument.

    Parameters
    ----------
    topology: str
        Graph topology: 'directed' or 'D' and 'undirected' or 'U'.

    elite: bool (default False)
        Indicates whether to use only the best ants for the pheromone update.

    Returns
    -------
    :function
        Algorithm optimised to work with the type of topology specified.
    """
    assert isinstance(topology, str), 'topology argument must be a string.'

    topology_ = topology.lower()

    if topology_ == 'd' or topology_ == 'directed':
        if elite:
            return updateDirEliteMMAS
        return updateDirMMAS
    elif topology_ == 'u' or topology_ == 'undirected':
        if elite:
            return updateUndEliteMMAS
        return updateUndMMAS
    else:
        assert False, 'Unrecognised topology parameter. Available options are: "directed" or "D" ' \
                      'and "undirected" or "U".'


def updateACS(topology: str):
    """
    Function that returns the Ant Colony System pheromone update strategy optimised for the
    graph representation to be explored.

    Parameters
    ----------
    topology: str
        Graph topology: 'directed' or 'D' and 'undirected' or 'U'.

    Returns
    -------
    :function
        Algorithm optimised to work with the type of topology specified.
    """
    assert isinstance(topology, str), 'topology argument must be a string.'

    topology_ = topology.lower()

    if topology_ == 'd' or topology_ == 'directed':
        return updateDirACS
    elif topology_ == 'u' or topology_ == 'undirected':
        return updateUndACS
    else:
        assert False, 'Unrecognised topology parameter. Available options are: "directed" or "D" ' \
                      'and "undirected" or "U".'


def updateDirLocalPher(ant: Ant, P: np.ndarray, decay: float, init_val: float):
    """
    Local pheromone update from Ant Colony System (ACS) based on

        Dorigo, M., Birattari, M., & Stutzle, T. (2006). Ant colony optimization. IEEE
        computational intelligence magazine, 1(4), 28-39.

    The local pheromone update is performed by all the ants after each construction step. Each ant
    applies

        P[i,j] = (1 - decay) * P[i,j] + decay * P[i,j]

    only to the last edge traversed.

    Parameters
    ----------
    ant: antco.Ant
        Ant instance.

    P: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone matrix to be updated.

    decay: float
        Decay value according to the equation presented in the Ant Colony System for performing
        the local pheromone update.

    init_val: float
        Pheromone initial value.

    Notes
    -----
    Function optimised for directed graphs.
    """
    i, j = ant.visited_nodes[-2:]
    P[i, j] = (1 - decay) * P[i, j] + decay * init_val


def updateUndLocalPher(ant: Ant, P: np.ndarray, decay: float, init_val: float):
    """
    Local pheromone update from Ant Colony System (ACS) based on

        Dorigo, M., Birattari, M., & Stutzle, T. (2006). Ant colony optimization. IEEE
        computational intelligence magazine, 1(4), 28-39.

    The local pheromone update is performed by all the ants after each construction step. Each ant
    applies

        P[i,j] = (1 - decay) * P[i,j] + decay * P[i,j]

    only to the last edge traversed.

    Parameters
    ----------
    ant: antco.Ant
        Ant instance.

    P: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone matrix to be updated.

    decay: float
        Decay value according to the equation presented in the Ant Colony System for performing
        the local pheromone update.

    init_val: float
        Pheromone initial value.

    Notes
    -----
    Function optimised for undirected graphs.
    """
    i, j = ant.visited_nodes[-2:]
    P[i, j] = (1 - decay) * P[i, j] + decay * init_val
    P[j, i] = (1 - decay) * P[j, i] + decay * init_val
