import numpy as np
import joblib
from antco import getRandomWalk, Ant, step


def test_undirected_random_walk():
    """ antco.colony.undirected_random_walk() unit testing"""
    np.random.seed(1997)

    heuristic = np.random.uniform(size=(6, 6)).astype(np.float64)
    pheromone = np.random.uniform(size=heuristic.shape).astype(np.float64)
    alpha, beta = 1.0, 1.0
    max_lim = 10
    Q = 0.0001
    R = 0.0001

    adj_matrix = np.array([
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [0, 1, 0, 1, 0, 1],
        [1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0]], dtype=np.int8)

    possible_paths = [
        [0, 3, 2, 1, 5],
        [0, 3, 2, 5, 1],
        [0, 4, 3, 2, 5, 1],
        [0, 3, 4],
        [0, 4, 3, 2, 1, 5],
        [0, 1, 2, 5],
        [0, 1, 2, 3, 4],
        [0, 1, 5, 2, 3, 4]]

    for _ in range(1_000):
        current_path = np.array([0])
        path = getRandomWalk(initial_position=4,
                             current_path=current_path,
                             adjacency_matrix=adj_matrix,
                             heuristic=heuristic,
                             pheromone=pheromone,
                             alpha=alpha,
                             max_lim=max_lim,
                             Q=Q, R=R).tolist()

        assert path in possible_paths, 'ERROR: antco.colony.undirected_random_walk()'

    print('SUCCESSFUL TEST: antco.colony.undirected_random_walk()')


def test_directed_random_walk():
    """ antco.colony.directed_random_walk() unit testing """
    np.random.seed(1997)

    heuristic = np.random.uniform(size=(6, 6)).astype(np.float64)
    pheromone = np.random.uniform(size=heuristic.shape).astype(np.float64)
    alpha, beta = 1.0, 1.0
    max_lim = 10
    Q = 0.0001
    R = 0.0001

    # Initial position (4) -> Valid paths: [0, 3]
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]], dtype=np.int8)

    possible_paths = [
        [4, 3, 0, 1, 2],
        [4, 0, 1, 2],
        [4, 0, 1, 5, 2],
        [4, 3, 2],
        [4, 3, 0, 1, 5, 2]
    ]

    for _ in range(1_000):
        current_path = np.array([4], dtype=np.int64)
        path = getRandomWalk(initial_position=4,
                             current_path=current_path,
                             adjacency_matrix=adj_matrix,
                             heuristic=heuristic,
                             pheromone=pheromone,
                             alpha=alpha,
                             max_lim=max_lim,
                             Q=Q, R=R).tolist()
        assert path in possible_paths, 'ERROR: antco.colony.directed_random_walk()'

    print('SUCCESSFUL TEST: antco.colony.directed_random_walk()')


def undirected_step():
    """ UNDIRECTED:
            antco.colony.step()
            antco.colony.update_ants()
        unit testing.
    """
    np.random.seed(1997)

    graph_type = 'u'
    adjacency_matrix = np.ones(shape=(6, 6), dtype=np.int8)
    heuristic = np.random.uniform(size=adjacency_matrix.shape).astype(np.float64)
    pheromone = np.random.uniform(size=adjacency_matrix.shape).astype(np.float64)
    alpha, beta = 1.0, 1.0
    max_lim = 40
    n_ants = 3

    for n_jobs in range(1, 3):
        ants = [Ant(l_min=0, l_max=max_lim, graph_type=graph_type) for _ in range(n_ants)]

        if n_jobs == 1:
            paths = []
            for ant, seed in zip(ants, np.random.randint(9999, size=len(ants))):
                paths.append(step(
                    ant=ant, adjacency_matrix=adjacency_matrix, heuristic=heuristic,
                    pheromone=pheromone, alpha=alpha, beta=beta, exp_heuristic=True,
                    Q=0.0001, R=0.0001))
        else:
            paths = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(step)(
                    ant, adjacency_matrix, heuristic, pheromone, alpha, beta, True, 0.0001, 0.0001)
                for ant, seed in zip(ants, np.random.randint(9999, size=len(ants))))

    print('SUCCESSFUL TEST: antco.colony.step() [UNDIRECTED]')


def directed_step():
    """ DIRECTED:
            antco.colony.step()
            antco.colony.update_ants()
        unit testing.
    """
    np.random.seed(1997)

    graph_type = 'd'
    adjacency_matrix = np.ones(shape=(6, 6), dtype=np.int8)
    for _ in range(10):
        i, j = np.random.randint(6, size=2)
        adjacency_matrix[i, j] = 0

    heuristic = np.random.uniform(size=adjacency_matrix.shape).astype(np.float64)
    pheromone = np.random.uniform(size=adjacency_matrix.shape).astype(np.float64)
    alpha, beta = 1.0, 1.0
    max_lim = 40
    n_ants = 3

    for n_jobs in range(1, 3):
        ants = [Ant(l_min=0, l_max=max_lim, graph_type=graph_type) for _ in range(n_ants)]

        if n_jobs == 1:
            paths = []
            for ant, seed in zip(ants, np.random.randint(9999, size=len(ants))):
                paths.append(step(
                    ant=ant, adjacency_matrix=adjacency_matrix, heuristic=heuristic,
                    pheromone=pheromone, alpha=alpha, beta=beta, exp_heuristic=True,
                    Q=0.0001, R=0.0001))
        else:
            paths = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(step)(
                    ant, adjacency_matrix, heuristic, pheromone, alpha, beta, True, 0.0001, 0.0001)
                for ant, seed in zip(ants, np.random.randint(9999, size=len(ants))))

    print('SUCCESSFUL TEST: antco.colony.step() [DIRECTED]')


def test():
    test_directed_random_walk()
    test_undirected_random_walk()
    undirected_step()
    directed_step()
