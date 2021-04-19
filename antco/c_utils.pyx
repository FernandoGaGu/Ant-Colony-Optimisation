# Module that includes a series of  tools for different operations carried out in different
# modules.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.time cimport time

cimport numpy as np
import numpy as np

INT_DTYPE = np.int8
DOUBLE_DTYPE = np.float64

ctypedef np.int8_t INT_DTYPE_t
ctypedef np.float64_t DOUBLE_DTYPE_t


srand(time(NULL))


cpdef np.ndarray[DOUBLE_DTYPE_t, ndim=2] minMaxScaling(
        np.ndarray[DOUBLE_DTYPE_t, ndim=2] matrix, np.ndarray[INT_DTYPE_t, ndim=2] graph):
    """
    Function that scales the matrix received as an argument using the min-max algorithm:

        x[i, j] = (x[i, j] - min(x[i, j])) / (max(x[i, j]) - min(x[i, j]))

    The scaling of the values is carried out inplace.

    Parameters
    ----------
    matrix: np.ndarray[np.int64_t, ndim=2]
        Two-dimensional matrix to be scaled.
    
    graph: np.ndarray[np.int8_t, ndim=2]
        Two-dimensional matrix indicating the graph structure.
    
    
    Returns
    -------
    :np.ndarray[DOUBLE_DTYPE_t, ndim=2] 
        Scaled matrix.
            
    Notes
    -----
    If the min and max values are equal, numerical stability is not assured. Arguments matrix and
    graph must be square matrices.
    """
    cdef double diff, min_val = 999_999.99, max_val = -999_999.99
    cdef int i, j, n_nodes = graph.shape[0]
    cdef np.ndarray[DOUBLE_DTYPE_t, ndim=2] scaled_matrix = np.zeros(shape=(n_nodes, n_nodes),
                                                                     dtype=DOUBLE_DTYPE)

    # Get the minimum and maximum
    for i in range(n_nodes):
        for j in range(n_nodes):
            if graph[i, j] == 1:
                if matrix[i, j] > max_val:
                    max_val = matrix[i, j]
                if matrix[i, j] < min_val:
                    min_val = matrix[i, j]

    if min_val == max_val:  # Avoid zero division error
        min_val += 1e-06

    diff = max_val - min_val

    # Scale matrix
    for i in range(n_nodes):
        for j in range(n_nodes):
            if graph[i, j] == 1:
                scaled_matrix[i, j] = (matrix[i, j] - min_val) / diff

    return scaled_matrix


cpdef int rouletteWheel(double[:] probs):
    """
    Function that given an array of probabilities makes a roulette wheel to choose an index.
    In this function it is not necessary that the probabilities add up to exactly 1, yhe function 
    supports that the sum of probabilities is approximately 1.

    Parameters
    ----------
    probs: double[:]
        Array of probabilities used to select the indices.      


    Returns
    -------
    :int
        Index selected randomly according to the probabilities indicated.    
    """
    cdef float shoot = rand() / (RAND_MAX - 1.0)
    cdef int i, length = len(probs)
    cdef double cumulative_probs = 0.0, current_prob = 0.0

    for i in range(length):
        current_prob = probs[i] + cumulative_probs
        if shoot < current_prob:
            return i

        cumulative_probs += probs[i]

    return length - 1
