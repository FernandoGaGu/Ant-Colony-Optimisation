# Module containing the necessary functions to compute the computeMetrics that monitor the behaviour
# of the algorithms.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
cimport numpy as np
import numpy as np
cimport cython

INT8_DTYPE = np.int8
INT64_DTYPE = np.int64
DOUBLE_DTYPE = np.float64

ctypedef np.int8_t INT8_DTYPE_t
ctypedef np.int64_t INT64_DTYPE_t
ctypedef np.float64_t DOUBLE_DTYPE_t


@cython.boundscheck(False)
cdef void applyMinMaxToPher(double[:] min_max, np.ndarray[np.float64_t, ndim=1] pheromones,
                            np.ndarray[INT8_DTYPE_t, ndim=1] connections):
    """
    Method that obtains the minimum and maximum values of the pheromone array as long as there is a 
    1 in the connections array (i.e. the array representing the node's connections).
    
    
    Parameters
    ----------
    minMaxScaling double[:] 
        Array of two elements in which the first corresponds to the minimum value found and the 
        second to the maximum value found.
        
    pheromones: np.ndarray[np.float64_t, ndim=1]
        Pheromone matrix from which the minimum and maximum values are to be returned. The minimum 
        and maximum value will be considered taking into account only those positions in 
        connecitions having a value of 1. 
    
    connections: np.ndarray[np.int8_t, ndim=1]  
        Array defining the connections to be considered to compute the maximum and minimum. The 
        connections correspond to the indices whose value is 1.
    """
    cdef int dim = pheromones.shape[0]
    cdef int i
    min_max[0] = 9_999_999.0
    min_max[1] = -9_999_999.0

    for i in range(dim):
        if connections[i] == 1:
            if pheromones[i] < min_max[0]:
                min_max[0] = pheromones[i]
            if pheromones[i] > min_max[1]:
                min_max[1] = pheromones[i]


@cython.boundscheck(False)
cdef int countNodes(np.ndarray[np.float64_t, ndim=1] pheromones,
                    np.ndarray[INT8_DTYPE_t, ndim=1] connections, double limit):
    """
    Method that counts the number of nodes whose pheromones value is above the received limit.
    
    Parameters
    ----------
    pheromones: np.ndarray[np.float64_t, ndim=1]  
    
    connections: np.ndarray[np.int8_t, ndim=1]
    
    limit: double 
    
    Returns
    -------
    :int
        Number of nodes.
    """
    cdef int count = 0
    cdef int i, dim = pheromones.shape[0]

    for i in range(dim):
        if connections[i] == 1 and pheromones[i] >= limit:
            count += 1

    return count

cpdef int getBranchingFactor(np.ndarray[np.float64_t, ndim=2] pheromones,
                             np.ndarray[INT8_DTYPE_t, ndim=2] adj_matrix, double lambda_val):
    """
    Method that calculates the lambda-branching factor of the pheromone matrix for the specified 
    lambda value.
    
    Parameters
    ---------
    pheromones: np.ndarray[np.float64_t, ndim=2]
        Pheromones matrix.
    
    adj_matrix: np.ndarray[np.int8_t, ndim=2]
        Adjacency matrix defining the graph structure.
    
    lambda_val: double
        Lambda value.
     
    Returns
    -------   
    :int
         Number of edges.
    """
    cdef double min_max[2]
    cdef double limit
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] visited_nodes = np.zeros(shape=adj_matrix.shape[0], dtype=INT64_DTYPE)
    cdef int num_nodes = 0
    cdef int dim = pheromones.shape[0]
    cdef int i, j

    for i in range(dim):
        for j in range(dim):
            if adj_matrix[i, j] == 1 and visited_nodes[j] != 1:
                visited_nodes[j] = 1
                applyMinMaxToPher(min_max, pheromones[j, :], adj_matrix[j, :])
                limit = min_max[0] + lambda_val*(min_max[1] - min_max[0])
                num_nodes += countNodes(pheromones[j, :], adj_matrix[j, :], limit)

    return num_nodes

