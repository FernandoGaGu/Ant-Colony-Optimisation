"""
Same problem as presented in the sd_simple.py script but with more difficulty.
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
        if not ant.is_valid:
            return 0.0
        nodes = self.getVisitedNodes(ant)

        return objective(nodes, self._adj_matrix, self._diff)[0]


# Problem definition
NODES = 101
EDGES = 800
OPTIMAL_PATH_LENGTH = 17
NOISE = 5
MIN_NOISE_LENGTH = 10
MAX_NOISE_LENGTH = 20
SEED = 1997 + 1

problem = generate_problem(
    NODES, EDGES, OPTIMAL_PATH_LENGTH, NOISE, MIN_NOISE_LENGTH, MAX_NOISE_LENGTH, SEED)


# Display Gmin and Gmax values
#display_group_values(problem)

# Algorithm parameters
n_ants = 100
elite = 40
graph_type = 'undirected'
iterations = 2_500
tol = 500
evaporation = 0.01 * 2
alpha = 1.0
beta = 1.0
pheromone_init = 5.0
seed = 1997
n_jobs = 8
pheromone_update = {"strategy": "as", "weight": 0.00125 * 2, "graph_type": graph_type}
Q = 0.2
R = 0.2
accessory_node = True
path_limits = (0, OPTIMAL_PATH_LENGTH+1 if accessory_node else OPTIMAL_PATH_LENGTH)

# Create objectiveFunction function
obj_function = MaxGroupDiff(problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])

# Create ACO instance
colony = antco.ACO(
    n_ants=n_ants, graph=problem['adj_matrix'], heuristic=None,
    objective=obj_function, iterations=iterations, graph_type=graph_type, evaporation=evaporation,
    alpha=alpha, beta=beta, pheromone_init=pheromone_init, path_limits=path_limits,
    pheromone_update=pheromone_update, n_jobs=n_jobs, seed=seed, tol=tol, Q=Q, R=R,
    scaleScores=antco.tools.MinMaxScaler(max_val=1.0, max_historic=True))

# Pre-process ACO instance
antco.preproc.apply(colony, accessory_node=True)
#antco.preproc.apply(colony, scale_heuristic={'min_val': 0.0, 'max_val': 1.0})

print('\nACO', colony)

# Run algorithm
start = time.time()
report = antco.algorithm.bagOfAnts(colony, bag_size=elite)
end = time.time()
print('\nTotal time: %.5f' % (end - start))

solution = report.best_solution['ant'].visited_nodes
solution = np.sort(solution[1:]).tolist() if accessory_node else np.sort(solution).tolist()
solution_score = objective(solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])[0]
optimal_solution = np.sort(problem['selected_nodes']).tolist()
optimal_solution_score = objective(optimal_solution, problem['adj_matrix'], problem['Gmax'] - problem['Gmin'])[0]

print('Solution found (%.4f): %r\nOptim solution (%.4f): %r' %
      (solution_score, solution, optimal_solution_score, optimal_solution))


# Save convergence results
antco.graphics.branchingFactor(report, save_plot='./convergence/sd_BoA_BF.png')
antco.graphics.convergence(
    report, title='\nSolution %.4f (Optimal %.4f)\n' % (solution_score, optimal_solution_score),
    save_plot='./convergence/sd_BoA_Convergence.png')
plt.show()
