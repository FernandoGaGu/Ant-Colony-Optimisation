"""
The problem presented consists of finding the sub-graph that maximises the mean of the values in a
given group (denoted as Gmax) by minimising the mean of the values in another group (denoted as
Gmin), in essence maximising the difference between Gmax and Gmin subjected to a graph structure.

This script uses the basic version of the Ant Colony Optimization algorithm, assumes knowledge of
the optimal sub-graph size and demonstrates that the algorithm is able to find the optimal sub-graph.

The results have been saved in the ./convergence folder.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from problem_utils import generate_problem, display_group_values, is_connected
import sys; sys.path.append('..')
import antco
from antco import hybrid


def objective(nodes: list, adj_matrix: np.ndarray, diff: np.ndarray) -> float:
    """ Compute the difference of the subnetwork associated to the nodes received as argument
    between the group Gmax and the group Gmin (the difference has been precomputed in the variable
    diff) returning the average.  """
    subnetwork = np.zeros(adj_matrix.shape, dtype=np.int8)
    subnetwork[tuple(np.meshgrid(nodes, nodes))] = 1
    subnetwork = np.multiply(subnetwork, adj_matrix)
    coords = np.where(subnetwork == 1)

    return float(np.mean(diff[coords]))


class MaxGroupDiff(antco.optim.ObjectiveFunction):
    """ Objective function that maximises the network values in one group by minimising the values
    in the other group. """
    def __init__(self, adjacency_matrix: np.ndarray, diff: np.ndarray):
        self._adj_matrix = adjacency_matrix
        self._diff = diff

    def evaluate(self, ant: antco.Ant) -> float:
        nodes = self.getVisitedNodes(ant)

        return objective(nodes, self._adj_matrix, self._diff)


# ------------------------------------------------------------------------------------------------ #
# Problem definition
NODES = 101
EDGES = 800
OPTIMAL_PATH_LENGTH = 17
NOISE = 5
MIN_NOISE_LENGTH = 10
MAX_NOISE_LENGTH = 20
SEED = 1997 + 1
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
# Algorithm parameters
n_ants = 100
bag_size = 2
out_of_bag_size = 1
graph_type = 'undirected'
iterations = 75
evaporation = 0.05
alpha = 1.0
beta = 1.0
pheromone_init = 5.0
seed = 1997
n_jobs = 8
pheromone_update = {'strategy': 'mmas', 'elite': 3, 'weight': 0.4,
                    'limits': (0.05, 10.0), 'graph_type': graph_type}
accessory_node = True
Q = None
R = 0.05
# Greedy search parameters
greedy_n_ants = 20
greedy_depth = 1
greedy_jobs = 8
# ------------------------------------------------------------------------------------------------ #

problem = generate_problem(
    NODES, EDGES, OPTIMAL_PATH_LENGTH, NOISE, MIN_NOISE_LENGTH, MAX_NOISE_LENGTH, SEED)

optimal_solution = np.sort(problem['selected_nodes']).tolist()
optimal_solution_score = objective(
    optimal_solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])

# Display Gmin and Gmax values
#display_group_values(problem)


path_limits = (0, OPTIMAL_PATH_LENGTH+1 if accessory_node else OPTIMAL_PATH_LENGTH)

# Create objectiveFunction function
obj_function = MaxGroupDiff(problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])

# Create ACO instance
colony = antco.ACO(
    n_ants=n_ants, graph=problem['adj_matrix'], heuristic=problem['Gmax'] - problem['Gmin'],
    objective=obj_function, iterations=iterations, graph_type=graph_type, evaporation=evaporation,
    alpha=alpha, beta=beta, pheromone_init=pheromone_init, path_limits=path_limits,
    pheromone_update=pheromone_update, n_jobs=n_jobs, seed=seed, Q=Q, R=R,
    scaleScores=antco.tools.MinMaxScaler(max_val=1.0, max_historic=True))

# Create Metaheuristic
local_search = hybrid.greedy.GreedySearchOP(add_to_old=False,
    antco_objective=obj_function, best_ants=greedy_n_ants, depth=greedy_depth,
    adj_matrix=problem['adj_matrix'], objective=objective, n_jobs=greedy_jobs,
    objective_args={'adj_matrix': problem['adj_matrix'], 'diff': problem['Gmax'] - problem['Gmin']})

# Pre-process ACO instance
antco.preproc.apply(colony, accessory_node=True)
antco.preproc.apply(colony, scale_heuristic={'min_val': 0.0, 'max_val': 1.0})


# Run algorithm
start = time.time()
report = antco.algorithm.bagOfAnts(colony, metaheuristic=local_search, bag_size=bag_size,
                                   out_of_bag_size=out_of_bag_size)
end = time.time()
print('\nTotal time: %.5f' % (end - start))

solution = report.best_solution['ant'].visited_nodes
solution = np.sort(solution[1:]).tolist() if accessory_node else np.sort(solution).tolist()

# Check if the solution satisfies the constrains
assert is_connected(solution, problem['adj_matrix']), \
    'The solution obtained does not correspond to a fully connected subgraph.'

solution_score = objective(solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])

print('Solution found (%.4f): %r\nOptim solution (%.4f): %r' %
      (solution_score, solution, optimal_solution_score, optimal_solution))

# Save convergence results
# Save convergence results
antco.graphics.branchingFactor(report, save_plot='./convergence/sd_hybrid_MMAS_BF.png')
plt.show()
antco.graphics.convergence(
    report, title='\nSolution %.4f (Optimal %.4f)\n' % (solution_score, optimal_solution_score),
    save_plot='./convergence/sd_hybrid_MMAS_convergence.png')
plt.show()
