# Module that collects the possible preprocessing steps applicable to an antco.ACO instance prior
# to performing the optimisation.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from .aco import ACO
from .c_utils import minMaxScaling


def apply(aco_obj: ACO, **kwargs):
    """
    Function that applies the specified transformations (passed as a string) to the ACO instance.

    Parameters
    ----------
    aco_obj: antco.ACO
        Ant Colony Optimization (aka ACO) instance on which the transformations are to be applied.

    **kwargs
        Transformations to be applied to the ACO instance. Only the indicated transformations will
        be applied.

        scale_heuristic: dict
            Scales the heuristic matrix to a range of values defined by [min_val, max_val]

            >>> apply(aco_obj, scale_heuristic={'min_val': 1.0, 'max_val': 2.0})

        accessory_node: bool
            Indicates whether to add an accessory node densely connected to all nodes in the
            network. This way all the ants will start from this accessory node, optimising the
            initial positioning in the network.

            >>> apply(aco_obj, accessory_node=True)

    """
    if 'scale_heuristic' in kwargs:
        _scaleHeuristic(aco_obj, **kwargs['scale_heuristic'])

    if 'accessory_node' in kwargs and kwargs['accessory_node'] is True:
        _addAccessoryNode(aco_obj)


def _scaleHeuristic(aco_obj: ACO, min_val: float = 0.0, max_val: float = 1.0, **_):
    """
    Function that scales the heuristic matrix to a specific range defined by min_val and max_val.

    Parameters
    ----------
    aco_obj: ACO:
        ACO instance.

    min_val: float, default=0.0
        Min value.

    max_val: float, default=1.0
        Maximum value.
    """
    heuristic = aco_obj.heuristic
    heuristic = minMaxScaling(heuristic, aco_obj.graph)
    heuristic[aco_obj.graph == 1] = heuristic[aco_obj.graph == 1] * (max_val - min_val) + min_val
    aco_obj.heuristic = (heuristic, True)


def _addAccessoryNode(aco_obj: ACO):
    """ Method that adds an accessory node to the heuristics, pheromone and connectivity graph
    matrices. """
    def _add_node(matrix: np.ndarray):
        """ Adds an extra densely connected node to all nodes of the matrix in the last position. """
        new_shape = matrix.shape[0] + 1, matrix.shape[1] + 1
        new_matrix = np.empty(new_shape, dtype=matrix.dtype)

        # Copy matrix values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                new_matrix[i, j] = matrix[i, j]

        # Add node using the maximum value
        new_matrix[:, -1] = np.max(matrix)
        new_matrix[-1, :] = np.max(matrix)

        return new_matrix

    graph = _add_node(aco_obj.graph)
    heuristic = _add_node(aco_obj.heuristic)
    pheromones = _add_node(aco_obj.pheromones)

    assert graph.shape[0] == graph.shape[1] == heuristic.shape[0] == heuristic.shape[1] == pheromones.shape[0] == \
           pheromones.shape[1], 'Incorrect shapes adding accessory node.'

    # Accessory node will be in the last position
    aco_obj.accessory_node = graph.shape[0] - 1
    aco_obj.graph = (graph, True)
    aco_obj.heuristic = (heuristic, True)
    aco_obj.pheromones = (pheromones, True)
    aco_obj.objectiveFunction.accessory_node = True
