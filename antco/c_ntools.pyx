# Module that defines the functions used to traverse and operate on graph structures.
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
cdef unsigned int isPresent(int value, np.ndarray[INT64_DTYPE_t, ndim=1] current_path):
    """
    Function that returns 1 if the value received as an argument is isPresent in the array, 
    otherwise it will return 0.
    
    Parameters
    ----------
    value: int
    
    current_path: np.ndarray[np.int64_t, ndim=1]
    
    Returns
    -------
    unsigned int
        1 = True, 0 = False
    """
    cdef unsigned int is_present = 0
    cdef int i, length = current_path.shape[0]

    for i in range(length):
        if current_path[i] == value:
            is_present = 1
            return is_present

    return is_present


cpdef np.ndarray[INT8_DTYPE_t, ndim=2] toDirAdjMatrix(
        np.ndarray[INT64_DTYPE_t, ndim=1] adj_list, int n_nodes):
    """
    Method that converts an adjacency list into a binary adjacency matrix.
    
    Parameters
    ----------
    adj_list: np.ndarray[np.int64_t, ndim=1]
        List composed of integers with the visited nodes. The nodes must be in the order in which 
        they have been traversed by the ant.
    
    n_nodes: int
        Number of nodes that the adjacency matrix must have. This parameter must be consistent with
        the indexes indicated in the adjacency list.

    Returns
    -------
    :np.ndarray[np.int8_t, ndim=2]
        Binary adjacency matrix.

    Notes
    ------
    Optimized function for directed graphs.
    """
    cdef int n, to_node, from_node = adj_list[0]
    cdef int length = adj_list.shape[0]
    cdef np.ndarray[INT8_DTYPE_t, ndim=2] adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes),
                                                                      dtype=INT8_DTYPE)

    for n in range(1, length):
        to_node = adj_list[n]
        adjacency_matrix[from_node, to_node] = 1
        from_node = to_node

    return adjacency_matrix


cpdef np.ndarray[INT8_DTYPE_t, ndim=2] toUndAdjMatrix(
        np.ndarray[INT64_DTYPE_t, ndim=1] adj_list, int n_nodes):
    """
    Method that converts an adjacency list into a binary adjacency matrix.

    Parameters
    ----------
    adj_list: np.ndarray[np.int64_t, ndim=1]
        List composed of integers with the visited nodes. The nodes must be in the order in which 
        they have been traversed by the ant.
    
    n_nodes: int
        Number of nodes that the adjacency matrix must have. This parameter must be consistent with
        the indexes indicated in the adjacency list.

    Returns
    -------
    :np.ndarray[np.int8_t, ndim=2]
        Binary adjacency matrix.

    Notes
    ------
    Optimized function for undirected graphs.
    """
    cdef int n, to_node, from_node = adj_list[0]
    cdef int length = adj_list.shape[0]
    cdef np.ndarray[INT8_DTYPE_t, ndim=2] adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes),
                                                                     dtype=INT8_DTYPE)

    for n in range(1, length):
        to_node = adj_list[n]
        adjacency_matrix[from_node, to_node] = 1
        adjacency_matrix[to_node, from_node] = 1  # Undirected connection
        from_node = to_node

    return adjacency_matrix


cpdef np.ndarray[INT64_DTYPE_t, ndim=1] toDirAdjList(int init_pos,
                                                     np.ndarray[INT8_DTYPE_t, ndim=2] adj_matrix):
    """
    Method that converts an adjacency matrix into an adjacency list.
    
    Parameters
    ----------
    init_pos: int
        Starting position from which to traverse the graph.
    
    adj_matrix: np.ndarray[np.int8_t, ndim=2]
        Binary adjacency matrix indicating the connections between the nodes This matrix must be of
        the integer type and be square. No restriction check will be performed.
    
    
    Returns
    -------
    :np.ndarray[np.int64_t, ndim=1]
        Numpy ndarray composed of integers with the visited nodes. The nodes will be in the order 
        in which  they have been traversed by the ant.

    Notes
    ------
    Optimized function for directed networks.
    """
    cdef int i, idx = 0
    cdef unsigned int end = 1
    cdef int dim = adj_matrix.shape[0]
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] adj_list = np.zeros(shape=adj_matrix.shape[0], dtype=INT64_DTYPE)

    # Add initial position
    adj_list[idx] = init_pos
    idx += 1

    while end != 0:
        end = 0
        for i in range(dim):
            if adj_matrix[init_pos, i] == 1 and init_pos != i:
                adj_list[idx] = i
                init_pos = i
                end = 1
                idx += 1
                break

    return adj_list[:idx]


cpdef np.ndarray[INT64_DTYPE_t, ndim=1] toUndAdjList(int init_pos,
                                                     np.ndarray[INT8_DTYPE_t, ndim=2] adj_matrix):
    """
    Method that converts an adjacency matrix into an adjacency list.

    Parameters
    ----------
    init_pos: int
        Starting position from which to traverse the graph. This argument is only necessary for 
        compatibility with the rest of the framework functions.
    
    adj_matrix: np.ndarray[np.int8_t, ndim=2]
        Binary adjacency matrix indicating the connections between the nodes This matrix must be of
        the integer type and be square. No restriction check will be performed.
    
    Returns
    -------
    :np.ndarray[np.int64_t, ndim=1]
        Numpy ndarray composed of integers with the visited nodes. The nodes will be in the order 
        in which  they have been traversed by the ant.

    Notes
    ------
    Optimized function for undirected networks.
    """
    cdef int i, dim = adj_matrix.shape[0], idx = 1
    cdef unsigned int end = 1
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] adj_list = np.zeros(shape=dim, dtype=INT64_DTYPE)
    cdef np.ndarray[INT8_DTYPE_t, ndim=1] visited = np.zeros(shape=dim, dtype=INT8_DTYPE)

    # Set the initial node as visited
    visited[init_pos] = 1
    adj_list[0] = init_pos

    while end != 0:
        end = 0
        for i in range(dim):
            if adj_matrix[init_pos, i] == 1 and visited[i] != 1:
                adj_list[idx] = i
                visited[i] = 1
                init_pos = i
                end = 1
                idx += 1
                break

    return adj_list[:idx]


cpdef list getValidPiaths(int position, np.ndarray[INT64_DTYPE_t, ndim=1] current_path,
                         np.ndarray[INT8_DTYPE_t, ndim=2] adj_matrix):
    """
    Function that according to the binary adjacency matrix that has been traversed by an ant and
    considering the graph structure received as an argument returns a list of tuples with the
    possible paths that the ant can choose.

    Parameters
    ----------
    position: int
        Current position of the ant.
    
    current_path: np.ndarray[np.int64_t, ndim=1]
        Current path. Array with the nodes visited by the ant.
    
    adj_matrix: np.ndarray[np.int8_t, ndim=2]
        Binary adjacency matrix representing the network.
    
    Returns
    -------
    :list
       Array with possible nodes to choose from.
    """
    cdef int i
    cdef int n_nodes = adj_matrix.shape[0]
    cdef list paths = []

    for i in range(n_nodes):
        if adj_matrix[position, i] == 1 and isPresent(i, current_path) == 0:
            paths.append(i)

    return paths

