"""
Script that tests the ant optimisation framework for solving the traveller salesman problem with
48 cities. The minimal tour has length 33523. The results of the convergence have been stored
in the ./convergence/ directory as att48_basic.png.

This script employs a hybrid strategy combining the conventional ant optimisation algorithm with
an evolutionary strategy defined in the metaheuristics package.
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from problem_utils import parse
import sys; sys.path.append('..')
import antco
import antco.metaheuristic as meta


OPTIMAL_PATH = [
    1, 8, 38, 31, 44, 18, 7, 28, 6, 37, 19, 27, 17, 43, 30, 36, 46, 33, 20, 47, 21, 32, 39, 48, 5,
    42, 24, 10, 45, 35, 4, 26, 2, 29, 34, 41, 16, 22, 3, 23, 14, 25, 13, 11, 12, 15, 40, 9]
OPTIMAL_PATH = [val - 1 for val in OPTIMAL_PATH]


def objective(nodes, cost_matrix):
    cost_ = 0.0
    for i in range(1, len(nodes)):
        cost_ += cost_matrix[nodes[i - 1], nodes[i]]

    return cost_ * -1,


class MinDistance(antco.optim.ObjectiveFunction):
    """ Definition of the objectiveFunction function to be maximised. """
    def __init__(self, cost_matrix: np.ndarray):
        self._cost_matrix = cost_matrix

    def evaluate(self, ant: antco.Ant):
        """ Length of the route taken by the ant. """
        if not ant.is_valid:
            return 0.0
        return objective(ant.visited_nodes, self._cost_matrix)[0]  # Return fitness value


# Load distance matrix
distance_matrix = parse('data/att48.tsp')
n_nodes = distance_matrix.shape[0]

# Optimal solution
optimal_value = -1*objective(OPTIMAL_PATH, distance_matrix)[0]

# Create the graph that will be explored (Densely connected undirected graph)
adjacency_matrix = np.ones(shape=(n_nodes, n_nodes))
np.fill_diagonal(adjacency_matrix, 0)

# Algorithm parameters
#   ACO
n_ants = 100
graph_type = 'undirected'
iterations = 500
evaporation = 0.1
alpha = 1.0
beta = 1.0
pheromone_init = 1.0
pheromone_update = {'strategy': 'as', 'weight': 0.1, 'graph_type': graph_type}
seed = 1997
tol = 200
n_jobs = 4
#   EA
best_ants = 15
population_size = 75
crossover_prob = 0.8
mutation_prob = 0.2
individual_mutation_prob = 0.05
tournsize = 2
generations = 100
elite = 15
ea_n_jobs = 1

# Create objectiveFunction function
obj_function = MinDistance(distance_matrix)

# Create metaheuristic
metaheuristic = meta.ea.PermutationGA(
    antco_objective=obj_function, genetic_objective=objective, best_ants=best_ants,
    population_size=population_size, crossover_prob=crossover_prob, mutation_prob=mutation_prob,
    individual_mutation_prob=individual_mutation_prob, generations=generations, tournsize=tournsize,
    hof=elite, n_jobs=ea_n_jobs, genetic_objective_args={'cost_matrix': distance_matrix})

# Create ACO instance
colony = antco.ACO(
    n_ants=n_ants, graph=adjacency_matrix, heuristic=-1*distance_matrix, objective=obj_function,
    iterations=iterations, graph_type=graph_type, evaporation=evaporation, alpha=alpha,
    beta=beta, pheromone_init=pheromone_init, path_limits=(n_nodes-1, n_nodes), tol=tol,
    pheromone_update=pheromone_update, n_jobs=n_jobs, seed=seed,
    scaleScores=antco.tools.MinMaxScaler(max_val=2.0, max_historic=True))
print('Colony:', colony)

# Pre-process ACO instance
antco.preproc.apply(colony, scale_heuristic={'min_val': 0.0, 'max_val': 1.0})

# Run algorithm
print(f'\nNumber of cities: {n_nodes}\n')
start = time.time()
report = antco.algorithm.hybrid(colony, metaheuristic=metaheuristic)
end = time.time()
print('\nTotal time: %.5f' % (end - start))

# Evaluate best solution (The score given by the cost function is multiplied by -1 since the cost
# function will give us the negative value)
best_solution_distance = MinDistance(distance_matrix).evaluate(report.best_solution['ant']) * -1
print('\nLength of the path: %.4f (Optimal length: %.4f)\n' % (best_solution_distance, optimal_value))

# Save convergence results
antco.graphics.branchingFactor(report, save_plot='./convergence/att48_GA_BF.png')
plt.show()
antco.graphics.convergence(
    report, title='\nSolution %.4f (Optimal %.4f)\n' % (best_solution_distance, optimal_value),
    save_plot='./convergence/att48_GA_Convergence.png')
plt.show()
