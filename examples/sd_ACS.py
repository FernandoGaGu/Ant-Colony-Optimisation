"""
The problem presented consists of finding the sub-graph that maximises the mean of the values in a
given group (denoted as Gmax) by minimising the mean of the values in another group (denoted as
Gmin), in essence maximising the difference between Gmax and Gmin subjected to a graph structure.

This script uses the basic version of the Ant Colony Optimization algorithm, assumes knowledge of
the optimal sub-graph size and demonstrates that the algorithm is able to find the optimal sub-graph.

The results have been saved in the ./convergence folder as SD_basic_BF.png and
SD_basic_Convergence.png
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from problem_utils import generate_problem, display_group_values
import sys; sys.path.append('..')
import antco


def objective(nodes: list, adj_matrix: np.ndarray, diff: np.ndarray) -> tuple:
    """ Compute the difference of the subnetwork associated to the nodes received as argument
    between the group Gmax and the group Gmin (the difference has been precomputed in the variable
    diff) returning the average.  """
    subnetwork = np.zeros(adj_matrix.shape, dtype=np.int8)
    subnetwork[tuple(np.meshgrid(nodes, nodes))] = 1
    subnetwork = np.multiply(subnetwork, adj_matrix)
    coords = np.where(subnetwork == 1)

    return float(np.mean(diff[coords])),


class MaxGroupDiff(antco.optim.ObjectiveFunction):
    """ Objective function that maximises the network values in one group by minimising the values
    in the other group. """
    def __init__(self, adjacency_matrix: np.ndarray, diff: np.ndarray):
        self._adj_matrix = adjacency_matrix
        self._diff = diff

    def evaluate(self, ant: antco.Ant) -> float:
        nodes = self.getVisitedNodes(ant)

        return objective(nodes, self._adj_matrix, self._diff)[0]


# Problem definition
NODES = 120
EDGES = 800
OPTIMAL_PATH_LENGTH = 15
NOISE = 5
MIN_NOISE_LENGTH = 10
MAX_NOISE_LENGTH = 20
SEED = 1997 + 1

problem = generate_problem(
    NODES, EDGES, OPTIMAL_PATH_LENGTH, NOISE, MIN_NOISE_LENGTH, MAX_NOISE_LENGTH, SEED)

optimal_solution = np.sort(problem['selected_nodes']).tolist()
optimal_solution_score = objective(
    optimal_solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])[0]

# Display Gmin and Gmax values
#display_group_values(problem)

# Algorithm parameters
n_ants = 30
graph_type = 'undirected'
iterations = 500
evaporation = 0.05
alpha = 1.0
beta = 1.0
pheromone_init = 5.0
seed = 1997
n_jobs = 2
pheromone_update = {'strategy': 'acs', 'decay': 0.1, 'graph_type': graph_type, 'weight': 1.0}
accessory_node = True
path_limits = (0, OPTIMAL_PATH_LENGTH+1 if accessory_node else OPTIMAL_PATH_LENGTH)

# Create objectiveFunction function
obj_function = MaxGroupDiff(problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])

# Create ACO instance
colony = antco.ACO(
    n_ants=n_ants, graph=problem['adj_matrix'], heuristic=problem['Gmax'] - problem['Gmin'],
    objective=obj_function, iterations=iterations, graph_type=graph_type, evaporation=evaporation,
    alpha=alpha, beta=beta, pheromone_init=pheromone_init, path_limits=path_limits,
    pheromone_update=pheromone_update, n_jobs=n_jobs, seed=seed, Q=0.5,
    scaleScores=antco.tools.MinMaxScaler(max_val=2.0, max_historic=True))

# Pre-process ACO instance
antco.preproc.apply(colony, accessory_node=False)
antco.preproc.apply(colony, scale_heuristic={'min_val': 0.0, 'max_val': 1.0})

print('\nACO', colony)

# Run algorithm
start = time.time()
report = antco.algorithm.antColonySystem(colony)
end = time.time()
print('\nTotal time: %.5f' % (end - start))

solution = report.best_solution['ant'].visited_nodes
solution = np.sort(solution[1:]).tolist() if accessory_node else np.sort(solution).tolist()
solution_score = objective(solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])[0]

print('Solution found (%.4f): %r\nOptim solution (%.4f): %r' %
      (solution_score, solution, optimal_solution_score, optimal_solution))

# Save convergence results
antco.graphics.branchingFactor(report, save_plot='./convergence/sd_simple_BF.png')
antco.graphics.convergence(
    report, title='\nSolution %.4f (Optimal %.4f)\n' % (solution_score, optimal_solution_score),
    save_plot='./convergence/sd_simple_Convergence.png')
plt.show()
