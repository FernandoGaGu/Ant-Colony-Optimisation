import numpy as np
from antco import Ant, randomInit, fixedPositions


def test():
    seed = 1997
    np.random.seed(seed)

    l_min = 2
    l_max = 4

    adjacency_matrix = np.array([
        [0, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0]], dtype=np.int8)

    # Check constructors
    _ = Ant(l_min, l_max, 'd')
    _ = Ant(l_min, l_max, 'directed')
    _ = Ant(l_min, l_max, 'u')
    _ = Ant(l_min, l_max, 'undirected')

    # Check limits
    l_ant = Ant(l_min, l_max, 'd')
    assert l_ant.max_length == l_max and l_ant.min_length == l_min, \
        'ERROR: antco.ant.max_length or antco.ant.min_length'

    # Check initialization
    # Directed
    d_ant = Ant(l_min, l_max, 'd')
    d_ant.initAdjMatrix(n_nodes=adjacency_matrix.shape[0])
    d_ant.setInitialPosition(randomInit(adjacency_matrix))
    assert d_ant.initial_position == 5, 'ERROR: antco.ant.randomInit() (Directed)'
    assert len(d_ant.visited_nodes) == 1, 'ERROR: antco.ant.getVisitedNodes'
    # Undirected
    u_ant = Ant(l_min, l_max, 'u')
    u_ant.initAdjMatrix(n_nodes=adjacency_matrix.shape[0])
    u_ant.setInitialPosition(randomInit(adjacency_matrix))
    u_ant._adjacency_matrix[1, 3] = 1
    assert len(u_ant.visited_nodes) == 2, 'ERROR: antco.ant.getVisitedNodes (Undirected)'

    # Check assignment workflow
    # Directed
    d_ant = Ant(l_min, l_max, 'd')
    d_ant.setInitialPosition(0)
    d_ant.adj_matrix = np.array([
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.int8)

    assert np.all(d_ant.visited_nodes == [0, 3, 2, 4, 1, 5]), \
        'ERROR: antco.ant.adj_matrix or antco.ant.getVisitedNodes (Directed)'
    assert not d_ant.is_valid, 'ERROR: antco.ant.is_valid'

    d_ant.adj_matrix = np.array([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert np.all(d_ant.visited_nodes == [0, 4, 1]), \
        'ERROR: antco.ant.adj_matrix or antco.ant.getVisitedNodes (Directed)'

    # Undirected
    u_ant = Ant(l_min, l_max, 'u')
    u_ant.setInitialPosition(0)
    u_ant.adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.int8)

    assert np.all(u_ant.visited_nodes == [0, 1, 2]), \
        'ERROR: antco.ant.adj_matrix or antco.ant.getVisitedNodes (Undirected)'

    u_ant.adj_matrix = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0]], dtype=np.int8)
    assert len(u_ant.visited_nodes) == 1, \
        'ERROR: antco.ant.adj_matrix or antco.ant.getVisitedNodes (Undirected)'

    # Random initial position
    struct = np.random.randint(2, size=u_ant.adj_matrix.shape, dtype=np.int8)
    u_ant.setInitialPosition(randomInit(struct))
    assert u_ant.initial_position == 0, 'ERROR: antco.ant.randomInit'

    # Ant fixed positions
    ants = [Ant(l_min=l_min, l_max=l_max, graph_type='u') for _ in range(10)]
    fixedPositions(ants, struct)
    covered_positions = [ant.initial_position for ant in ants[:6]]
    assert np.sort(covered_positions).tolist() == [0, 1, 2, 3, 4, 5], \
        'ERROR: antco.ant.fixedPositions'

    print('SUCCESSFUL TEST: antco.ant.Ant')
