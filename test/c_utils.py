import numpy as np
from antco import minMaxScaling, rouletteWheel


def test_min_max():
    """ antco.utils.minMaxScaling() testing unit """
    def round(value):
        return np.round(value, decimals=4)

    matrix = np.array([
        [4, 3, 4, 0],
        [1, 2, 4, 2],
        [1, 5, 4, 6],
        [1, 5, 4, 6]], dtype=float)

    graph = np.array([
        [0, 1, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 1],
        [0, 0, 0, 0]], dtype = np.int8)

    numbers_to_scale = np.array([3, 0, 2, 1, 6])

    scaled_matrix = (matrix - np.min(numbers_to_scale)) / (np.max(numbers_to_scale) - np.min(numbers_to_scale))
    scaled_matrix_test = minMaxScaling(matrix, graph)

    assert round(scaled_matrix[0, 1]) == round(scaled_matrix_test[0, 1]) and \
           round(scaled_matrix[0, 3]) == round(scaled_matrix_test[0, 3]) and \
           round(scaled_matrix[2, 0]) == round(scaled_matrix_test[2, 0]) and \
           round(scaled_matrix[2, 3]) == round(scaled_matrix_test[2, 3]), \
        'FAILED TEST: antco.utils.minMaxScaling()'

    print('SUCCESSFUL TEST: antco.utils.minMaxScaling()')


def test_roulette_wheel():
    """ antco.utils.rouletteWheel() testing unit """
    np.random.seed(1997)
    probs = np.array([0.0, 1.0, 0.0, 0.0, 0.0])

    index = rouletteWheel(probs)
    assert index == 1, 'FAILED TEST: antco.utils.rouletteWheel()'

    probs = np.zeros(shape=1000)
    probs[-1] = 1.0

    index = rouletteWheel(probs)

    assert index == 999, 'FAILED TEST: antco.utils.rouletteWheel()'

    print('SUCCESSFUL TEST: antco.utils.rouletteWheel()')


def test():
    test_min_max()
    test_roulette_wheel()

