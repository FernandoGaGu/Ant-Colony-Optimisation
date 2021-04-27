import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import re


def euclidean(a, b):
    """ Calculates the Euclidean distance between two vectors. """
    return np.linalg.norm(np.array(a) - np.array(b))


def parse(file: str) -> np.ndarray:
    """ Reads data from a standard .tsp file. """
    with open(file) as input_file:
        f = input_file.read().split('\n')
    coords = []
    for elem in f:
        if elem == 'EOF':
            break
        init_line = elem.split(' ')[0]
        if not init_line.isnumeric():
            continue
        #split_elms = [val for val in elem.split(' ')[1:] if re.match(r'^-?\d+(?:\.\d+)$', val)]

        x, y = elem.split(' ')[1:]
        coords.append((float(x), float(y)))

    distance_matrix_ = np.zeros(shape=(len(coords), len(coords)))

    for i in range(distance_matrix_.shape[0]):
        for j in range(distance_matrix_.shape[1]):
            if not i == j:
                distance_matrix_[i, j] = euclidean(coords[i], coords[j])
            else:
                distance_matrix_[i, j] = 0.0

    return distance_matrix_


def is_connected(nodes: list or np.ndarray, adj_matrix: np.ndarray) -> bool:
    """
    Method that checks whether the nodes received as a list form a sub-network without any isolated
    nodes within the network defined by the adjacency matrix.

    Parameters
    ----------
    nodes: np.ndarray or list
        Sub-network nodes.

    adj_matrix: np.ndarray
        Adjacency matrix defining the graph structure.

    Returns
    -------
    :bool
        True if the sub-network is connected, False otherwise.
    """
    graph = nx.from_numpy_array(adj_matrix)
    subgraph = graph.subgraph(nodes)

    return nx.is_connected(subgraph)


def generate_problem(nodes: int, edges: int, optimal_path_length: int, noise: int,
                     min_noise_length: int, max_noise_length: int, seed: int):
    """
    Function to generate an optimal sub-graph detection problem.

    Parameters
    ----------
    nodes: int
        Number of nodes of the network to be explored.
    
    edges: int
        Number of edges in the network to be explored.

    optimal_path_length: int
        Number of nodes in the optimal sub-network.

    noise: int
        Noise added to the problem. This number determines the number of random subnetworks (with
        less differences than the principal one) added to the problem.

    min_noise_length: int
        Minimum number of nodes in the noise subnetworks.

    max_noise_length: int
        Maximum number of nodes in the noise subnetworks.

    seed: int
        Random seed.

    Returns
    -------
    :dict
        :key "graph": nx.Graph
            Graph to be explored.
        :key "adj_matrix": np.ndarray (nodes, nodes), dtype=np.int8
            Graph to be explored (as adjacency matrix).
        :key "optimal_subgraph": np.ndarray (nodes, nodes), dtype=np.int8
            Optimal subgraph (as adjacency matrix).
        :key "selected_nodes": np.ndarray(optimal_path_length), dtype=int
            Optimal subgraph nodes.
        :key "Gmax": np.ndarray (nodes, nodes), dtype=np.float64
            Group 1 values.
        :key "Gmin": np.ndarray (nodes, nodes), dtype=np.float64
            Group 2 values.
    """
    problem = dict()

    random.seed(seed)
    np.random.seed(seed)

    # Generate a random graph
    graph = nx.dense_gnm_random_graph(n=nodes, m=edges, seed=seed)
    adj_matrix = np.array(nx.adjacency_matrix(graph).todense(), dtype=np.int8)

    # Select a random optimal sub-graph
    selected_nodes = np.random.choice(
        [n for n in range(nodes)], size=optimal_path_length, replace=False)

    valid_subgraph = False
    saveguard = 9999
    count = 0
    while not valid_subgraph:
        if is_connected(selected_nodes, adj_matrix):
            valid_subgraph = True
        else:
            selected_nodes = np.random.choice(
                [n for n in range(nodes)], size=optimal_path_length, replace=False)
        count += 1
        if count == saveguard:
            assert False, 'impossible to obtain a connected subgraph from the network, consider ' \
                          'increasing the number of connections.'

    subgraph = np.zeros(shape=adj_matrix.shape, dtype=np.int8)
    subgraph[tuple(np.meshgrid(selected_nodes, selected_nodes))] = 1
    subgraph = np.multiply(subgraph, adj_matrix).astype(np.int8)

    # Generate group values
    Gmax = np.multiply(np.random.uniform(low=0.2, high=0.4, size=adj_matrix.shape), adj_matrix)
    Gmin = np.multiply(np.random.uniform(low=-0.2, high=0.2, size=adj_matrix.shape), adj_matrix)

    # max(Gmax) = 1.2; min(Gmax) =  0.8
    # max(Gmin) = -0.4; min(Gmin) = -1.0
    # Max difference: 2.2
    # Min difference: 1.2
    variation = np.multiply(np.random.uniform(low=0.6, high=0.8, size=adj_matrix.shape), subgraph)
    Gmax += variation
    Gmin -= variation

    # Add noise
    possible_nodes = list(set([n for n in range(nodes)]) - set(selected_nodes))
    for i in range(noise):
        noise_nodes = np.random.choice(
            possible_nodes, size=random.randint(min_noise_length, max_noise_length), replace=False)
        possible_nodes = list(set(possible_nodes) - set(noise_nodes))
        if len(possible_nodes) < max_noise_length:
            print('Too much noise has been introduced, the added noise may generate a better '
                  'sub-network than optimal sub-network. The noise factor has been reduced to %d' % i)
            break

        random_subnetwork = np.zeros(shape=adj_matrix.shape, dtype=np.int8)
        random_subnetwork[tuple(np.meshgrid(noise_nodes, noise_nodes))] = 1
        random_subnetwork = np.multiply(random_subnetwork, adj_matrix).astype(np.int8)
        # max(Gmax) = 0.6; min(Gmax) =  0.2
        # max(Gmin) = 0.2; min(Gmin) = -0.4
        # Max difference: 1.0
        # Min difference: 0.0
        variation = np.multiply(np.random.uniform(low=0.0, high=0.2, size=adj_matrix.shape),
                                random_subnetwork)
        Gmax += variation
        Gmin -= variation

    # Generate symmetric matrices
    Gmax = (Gmax + Gmax.T) / 2
    Gmin = (Gmin + Gmin.T) / 2


    problem['graph'] = graph
    problem['adj_matrix'] = adj_matrix
    problem['optimal_subgraph'] = subgraph
    problem['selected_nodes'] = selected_nodes
    problem['Gmax'] = Gmax
    problem['Gmin'] = Gmin

    return problem


def display_group_values(problem: dict):
    max_Gmax = np.max(problem['Gmax'])
    max_Gmin = np.max(problem['Gmin'])
    max_val = max([max_Gmax, max_Gmin])

    min_Gmax = np.min(problem['Gmax'])
    min_Gmin = np.min(problem['Gmin'])
    min_val = min([min_Gmax, min_Gmin])

    ax = plt.axes()
    sns.heatmap(problem['Gmax'], ax=ax, vmin=min_val, vmax=max_val)
    ax.set_title('Group (max)')
    plt.show()

    ax = plt.axes()
    sns.heatmap(problem['Gmin'], ax=ax, vmin=min_val, vmax=max_val)
    ax.set_title('Group (min)')
    plt.show()
