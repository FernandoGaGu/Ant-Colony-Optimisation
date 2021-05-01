# Module that groups the greedy search strategies that can be incorporated as a hybrid strategy in
# the algorithms.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import joblib
import networkx as nx
from .base import MetaHeuristic
from ..optim import ObjectiveFunction
from ..c_ntools import getValidPaths


class GreedySearchOP(MetaHeuristic):
    """
    Class defining a greedy search strategy for the selection of optimal paths.This class allows to
    optimise the path given by the ants following the following algorithm:

    ----------------------------------------------------------------------------------------------
        Input: G (Graph defining the network structure)
        Input: max_depth (Maximum depth of the exploration)
        Input: ant (Ant given by the ACO algorithm)

        1. solution <- ant.visited_nodes
        2. solution_score <- evaluate(solution)
        3. depth = 0
        4. While depth < max_depth:
        5.     improvement = False
               # Get those nodes that when removed do not generate two disconnected sub-networks
        6.     nodes <- getReplaceable(solution)
        7.     for node in nodes:
        8.         new_solution <- remove(solution, node)  # Eliminate node from solution
                   # Nodes that can be added to the network according to the network structure
        9.         available_nodes <- getValidNodes(new_solution, G)
        10.         for new_node in available_nodes:
        11.             new_solution_temp <- add(new_solution, new_node)
        12.            new_score <- evaluate(new_solution_temp)
        13.            if new_score > solution_score:
        14.                solution <- new_solution_temp
        15.                solution_score <- new_score
                           improvement = True
        16.    if not improvement: break
        17.    depth++
        18. Process and return solution
    ----------------------------------------------------------------------------------------------

    Important note: This class is not of general purpose, for certain types of problems the
    strategy may not make sense. Also, depending on the network to be explored, it may be
    inefficient.

    Important note (2): See parameter add_to_old.

    Parameters
    ----------
    antco_objective: antco.optim.ObjectiveFunction
        Objective function defined using the antco.optim.ObjectiveFunction interface.

    best_ants: int
        Number of the best ants to be passed to the hybrid strategy.

    objective: callable
        Objective function that will receive each solution (encoded as a list of integers without
        repetition) and will return a scalar value evaluating the solution.

    adj_matrix: np.ndarray
        Adjacency matrix defining the network to be explored.

    n_jobs: int, default=1
        Number of processes running in parallel.

    objective_args: dict, default=None
        If the objective function requires additional parameters these must be passed to the
        constructor in the form of a dictionary where the name of the parameter received in the
        objective function must correspond to the key and the value passed to the value.

    add_to_old: bool, default=False
        Parameter indicating whether to add the new solution to the set of ants or to only consider
        the number of ants returned by this strategy for the update of the pheromone matrix. In the
        latter case, the default behaviour, if the number of ants exploring the network is for
        example 100 and the parameter best_ants is 10, the number of ants that will be available
        for the pheromone matrix update will be 10. If the parameter is specified as True, the
        number of ants available will be 100 + 10.
    """
    def __init__(self, antco_objective: ObjectiveFunction, best_ants: int, depth: int,
                 objective: callable, adj_matrix: np.ndarray, n_jobs: int = 1,
                 objective_args: dict = None, add_to_old: bool = False):
        super(GreedySearchOP, self).__init__(antco_objective, add_to_old)

        self._best_ants = best_ants
        self._depth = depth
        self._adj_matrix = adj_matrix
        self._objective = objective
        self._objective_args = {} if objective_args is None else objective_args
        self._n_jobs = n_jobs

    def __repr__(self):
        return f'GreedySearchOP(best_ants={self._best_ants}, depth={self._depth}, ' \
               f'n_jobs={self._n_jobs})'

    def __str__(self):
        return self.__repr__()

    def optimise(self, ants: list, scores: list) -> tuple:
        # Get the index associated with the best ants
        best_ants_indices = np.argsort(scores)[::-1][:self._best_ants]
        improved_solutions = []
        improved_scores = []

        if self._n_jobs > 1:
            solution_score = joblib.Parallel(n_jobs=self._n_jobs, backend='loky')(
                joblib.delayed(greedySearch)(
                    self.evaluate.getVisitedNodes(ants[idx]), self._depth, self._adj_matrix,
                    self._objective, self._objective_args, scores[idx]) for idx in best_ants_indices)

            for (new_solution, new_score) in solution_score:
                improved_solutions.append(new_solution)
                improved_scores.append(new_score)

        else:  # Serial execution
            for idx in best_ants_indices:
                new_solution, new_score = greedySearch(
                    nodes=self.evaluate.getVisitedNodes(ants[idx]),
                    depth=self._depth,
                    adj_matrix=self._adj_matrix,
                    objective=self._objective,
                    objective_args=self._objective_args,
                    best_score=scores[idx])

                improved_solutions.append(new_solution)
                improved_scores.append(new_score)

        # Create ants
        improved_ants = [ants[0].new() for _ in range(self._best_ants)]
        for idx, solution in enumerate(improved_solutions):
            improved_ants[idx].visited_nodes = solution

        return improved_ants, improved_scores


def greedySearch(nodes: list or np.ndarray, depth: int, adj_matrix: np.ndarray,
                 objective: callable, objective_args: dict, best_score: float):
    """
    Greedy search strategy used in GreedySearchOP.

    Parameters
    ----------
    nodes: list or np.ndarray (solution_nodes)
        Nodes which make up the solution to be improved.

    depth: int
        Maximum depth of the solution tree explored.

    adj_matrix: np.ndarray (nodes, nodes)
        Adjacency matrix defining the structure of the network to be explored.

    objective: callable
        Objective function to be maximised used for the evaluation of solutions.

    objective_args: dict
        Additional parameters that will be passed to the objective function.

    best_score: float
        Initial score associated to the initial solution.

    Returns
    -------
    :tuple
        Tuple where the first element correspond to the improved solution or the solution received
        if it could not be improved, and the second argument to the score associated with the
        solution.
    """
    graph = nx.from_numpy_array(adj_matrix)

    best_solution = None
    current_depth = 0
    while current_depth < depth:
        if current_depth == 0:
            nodes_to_explore = [n for n in nodes]
        else:
            if best_solution is None: break
            nodes_to_explore = best_solution

        # Get those nodes that when removed do not generate two disconnected sub-networks
        repl_nodes = getReplaceable(nodes_to_explore, graph)

        # Replace each possible node
        for n1 in repl_nodes:
            # Solution in which one of the nodes has been removed
            pruned_solution = [node for node in nodes_to_explore]
            pruned_solution.pop(pruned_solution.index(n1))

            # Get a list of all nodes that can be added to the sub-network according to the network
            # structure
            substitutes = []
            for n2 in pruned_solution:
                substitutes.append(
                    getValidPaths(n2, np.array(pruned_solution), adj_matrix))

            # Getting the unique node
            substitutes = list(set([e for lt in substitutes for e in lt if not e == n1]))

            for n2 in substitutes:
                new_score = objective(pruned_solution + [n2], **objective_args)
                if new_score > best_score:
                    best_solution = pruned_solution + [n2]
                    best_score = new_score
        current_depth += 1

    if best_solution is None:
        return nodes, best_score

    return best_solution, best_score


def getReplaceable(nodes: np.ndarray, graph: nx.Graph) -> list:
    """
    Function that returns the nodes from the nodes argument, that can be removed from the network
    without generating two disconnected sub-networks.

    Parameters
    ----------
    nodes: np.ndarray
        Nodes composing the sub-graph whose nodes are to be examined.

    graph: nx.Graph
        Graph defining the structure of the network to be explored.

    Returns
    -------
    :list
        Nodes that can be replaced without generating two disconnected sub-networks.
    """
    replaceable_nodes = []
    for node in nodes:
        selected_nodes = [n for n in nodes if n != node]
        subgraph = graph.subgraph(selected_nodes)
        if nx.is_connected(subgraph):
            replaceable_nodes.append(node)

    return replaceable_nodes

