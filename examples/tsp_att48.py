"""
ATT48 is a set of 48 cities (US state capitals) from TSPLIB. The minimal tour has length 33523.

Data from:

    https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from problem_utils import parse
import sys; sys.path.append('..')
import antco


OPTIMAL_PATH = [
    1, 8, 38, 31, 44, 18, 7, 28, 6, 37, 19, 27, 17, 43, 30, 36, 46, 33, 20, 47, 21, 32, 39, 48, 5,
    42, 24, 10, 45, 35, 4, 26, 2, 29, 34, 41, 16, 22, 3, 23, 14, 25, 13, 11, 12, 15, 40, 9]
OPTIMAL_PATH = [val - 1 for val in OPTIMAL_PATH]


class MinDistance(antco.optim.ObjectiveFunction):
    """ Definition of the objectiveFunction function to be maximised. """
    def __init__(self, cost_matrix: np.ndarray, objectiveFunc: callable):
        self._cost_matrix = cost_matrix
        self._objective = objectiveFunc

    def evaluate(self, ant: antco.Ant):
        """ Length of the route taken by the ant. """
        if not ant.is_valid:
            return 0.0

        return self._objective(self.getVisitedNodes(ant), self._cost_matrix)


def objective(nodes: list, cost_matrix: np.ndarray):
    cost_ = 0.0
    for i in range(1, len(nodes)):
        cost_ += cost_matrix[nodes[i - 1], nodes[i]]

    return cost_ * -1  # Transform to minimisation problem


# Load distance matrix
distance_matrix = parse('data/att48.tsp')
n_nodes = distance_matrix.shape[0]

# Optimal solution
optimal_value = -1*objective(OPTIMAL_PATH, distance_matrix)

# Create the graph that will be explored (Densely connected undirected graph)
adjacency_matrix = np.ones(shape=(n_nodes, n_nodes))
np.fill_diagonal(adjacency_matrix, 0)

# Algorithm parameters
n_ants = 100
graph_type = 'undirected'
iterations = 1_000
evaporation = 0.1
alpha = 1.0
beta = 1.0
pheromone_init = 5.0
seed = 1997
tol = 200
n_jobs = 8
pheromone_update = {'strategy': 'mmas', 'limits': (0.05, 5.0), 'graph_type': graph_type}
aco_obj = MinDistance(distance_matrix, objective)

# Create ACO instance
colony = antco.ACO(
    n_ants=n_ants, graph=adjacency_matrix, heuristic=(-1*distance_matrix), objective=aco_obj,
    iterations=iterations, graph_type=graph_type, evaporation=evaporation, alpha=alpha,
    beta=beta, pheromone_init=pheromone_init, path_limits=(n_nodes-1, n_nodes),
    pheromone_update=pheromone_update, n_jobs=n_jobs, seed=seed, tol=tol,
    scaleScores=antco.tools.MinMaxScaler(max_val=2.0, max_historic=True))

antco.preproc.apply(colony, scale_heuristic={'min_val': 0.0, 'max_val': 1.0})

# Run algorithm
print(f'\nNumber of cities: {n_nodes}\n')
start = time.time()
report = antco.algorithm.basic(colony)
end = time.time()
print('\nTotal time: %.5f' % (end - start))

# Evaluate best solution (The score given by the cost function is multiplied by -1 since the cost
# function will give us the negative value)
best_solution_distance = aco_obj.evaluate(report.best_solution['ant']) * -1
print('\nLength of the path: %.4f (Optimal length: %.4f)\n' % (best_solution_distance, optimal_value))

# Save convergence results
antco.graphics.branchingFactor(report, save_plot='./convergence/att48_BF.png')
plt.show()
antco.graphics.convergence(
    report, title='\nSolution %.4f (Optimal %.4f)\n' % (best_solution_distance, optimal_value),
    save_plot='./convergence/att48_Convergence.png')
plt.show()
