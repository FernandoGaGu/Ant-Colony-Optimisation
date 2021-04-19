import numpy as np
from antco import (
    toDirAdjMatrix,
    toUndAdjMatrix,
    toDirAdjList,
    toUndAdjList,
    getValidPaths)


def test_to_directed_adjacency_matrix():
    """ antco.ntools.toDirAdjMatrix() testing unit """
    expected = [
        np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.int8),
        np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.int8)]

    arguments = [np.array([0, 1, 2, 4, 3]), np.array([0, 1, 2, 3])]

    for adj_matrix, adj_list in zip(expected, arguments):

        output = toDirAdjMatrix(adj_list, adj_matrix.shape[0])

        assert np.all(output == adj_matrix), 'FAILED TEST: antco.ntools.toDirAdjMatrix()'

    print('SUCCESSFUL TEST: antco.ntools.toDirAdjMatrix()')


def test_to_undirected_adjacency_matrix():
    """ antco.c_ntools.toUndAdjMatrix() testing unit """
    expected = [
        np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0]], dtype=np.int8),
        np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]], dtype=np.int8)]

    arguments = [np.array([0, 1, 2, 4, 3]), np.array([0, 1, 2, 3])]

    for adj_matrix, adj_list in zip(expected, arguments):

        output = toUndAdjMatrix(adj_list, adj_matrix.shape[0])

        assert np.all(output == adj_matrix), 'FAILED TEST: antco.ntools.toUndAdjMatrix()'

    print('SUCCESSFUL TEST: antco.ntools.toUndAdjMatrix()')


def test_to_directed_adjacency_list():
    """ antco.ntools.toDirAdjList() testing unit """
    arguments = [
        (0, np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0]], dtype=np.int8)),
        (4, np.array([
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]], dtype=np.int8))]

    expected = [np.array([0, 1, 2, 4, 3]), np.array([4, 0, 1, 2, 3])]

    for args, adj_list in zip(arguments, expected):
        output = toDirAdjList(*args)

        assert np.all(output == adj_list), 'FAILED TEST: antco.ntools.toDirAdjList()'

    print('SUCCESSFUL TEST: antco.ntools.toDirAdjList()')


def test_to_undirected_adjacency_list():
    """ antco.ntools.toUndAdjList() testing unit """
    arguments = [
        (0, np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0]], dtype=np.int8)),
        (4, np.array([
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]], dtype=np.int8))]

    expected = [np.array([0, 1, 2, 4, 3]), np.array([4, 0, 1, 2])]

    for args, adj_list in zip(arguments, expected):
        output = toUndAdjList(*args)
        assert np.all(output == adj_list), 'FAILED TEST: antco.ntools.toUndAdjList()'

    print('SUCCESSFUL TEST: antco.ntools.toUndAdjList()')


def test_undirected_valid_paths():
    """ antco.ntools.getValidPaths() (undirected) testing unit """
    adj_matrix = np.array([
        [0, 1, 0, 1, 1, 0],
        [1, 0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1],
        [1, 1, 0, 0, 1, 1],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 0]], dtype=np.int8)

    # Initial position (0) -> Valid paths: (1, 3, 4)
    current_path = np.array([0], dtype=np.int64)
    choices = getValidPaths(0, current_path, adj_matrix)
    assert choices == [1, 3, 4], 'FAILED TEST: antco.ntools.test_valid_paths() (undirected)'

    # Initial position (2) -> Valid paths: (1, 5)
    current_path = np.array([0], dtype=np.int64)
    choices = getValidPaths(2, current_path, adj_matrix)
    assert choices == [1, 5], 'FAILED TEST: antco.ntools.test_valid_paths() (undirected)'

    # Initial position (2) -> Valid paths: (1) [5 visited]
    current_path = np.array([5], dtype=np.int64)
    choices = getValidPaths(2, current_path, adj_matrix)
    assert choices == [1], 'FAILED TEST: antco.ntools.test_valid_paths() (undirected)'

    print('SUCCESSFUL TEST: antco.ntools.test_valid_paths() (undirected)')


def test_directed_valid_paths():
    """ antco.ntools.getValidPaths() (directed) testing unit """
    # Initial position (0) -> Valid paths: (1)
    adj_matrix = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0]], dtype=np.int8)

    current_path = np.array([0], dtype=np.int64)
    choices = getValidPaths(0, current_path, adj_matrix)
    assert choices == [1], 'FAILED TEST: antco.ntools.getValidPaths() (directed)'

    current_path = np.array([0], dtype=np.int64)
    choices = getValidPaths(1, current_path, adj_matrix)
    assert choices == [2, 5], 'FAILED TEST: antco.ntools.getValidPaths() (directed)'

    current_path = np.array([5], dtype=np.int64)
    choices = getValidPaths(1, current_path, adj_matrix)
    assert choices == [2], 'FAILED TEST: antco.ntools.getValidPaths() (directed)'

    current_path = np.array([5, 2], dtype=np.int64)
    choices = getValidPaths(1, current_path, adj_matrix)
    assert choices == [], 'FAILED TEST: antco.ntools.getValidPaths() (directed)'

    print('SUCCESSFUL TEST: antco.ntools.test_valid_paths() (directed)')


def test():
    test_to_directed_adjacency_matrix()
    test_to_undirected_adjacency_matrix()
    test_to_directed_adjacency_list()
    test_to_undirected_adjacency_list()
    test_undirected_valid_paths()
    test_directed_valid_paths()

