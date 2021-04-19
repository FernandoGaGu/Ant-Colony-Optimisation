import numpy as np
from antco import getBranchingFactor


def test_directed_branching_factor():
    """ antco.computeMetrics.getBranchingFactor() (DIRECTED)"""
    np.random.seed(1997)

    n_nodes = 6
    lambda_val = 0.5

    pheromones = np.random.uniform(size=(n_nodes, n_nodes))

    adj_matrix = np.array([
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0]], dtype=np.int8)

    val = getBranchingFactor(pheromones, adj_matrix, lambda_val)
    assert val == 7, 'FAILED TEST: antco.computeMetrics.getBranchingFactor() (directed)'

    print('SUCCESSFUL TEST: antco.computeMetrics.getBranchingFactor() (directed)')



def test_undirected_branching_factor():
    """ antco.computeMetrics.getBranchingFactor() (UNDIRECTED)"""
    np.random.seed(1997)

    n_nodes = 6
    lambda_val = 0.5

    pheromones = np.random.uniform(size=(n_nodes, n_nodes))

    adj_matrix = np.array([
        [0, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1, 1],
        [1, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0]], dtype=np.int8)

    for i in range(adj_matrix.shape[0]):
        for j in range(i + 1, adj_matrix.shape[1]):
            pheromones[i, j] = pheromones[j, i]

    val = getBranchingFactor(pheromones, adj_matrix, lambda_val)
    assert val == 10, 'FAILED TEST: antco.computeMetrics.getBranchingFactor() (undirected)'

    print('SUCCESSFUL TEST: antco.computeMetrics.getBranchingFactor() (undirected)')

def test():
    test_directed_branching_factor()
    test_undirected_branching_factor()
