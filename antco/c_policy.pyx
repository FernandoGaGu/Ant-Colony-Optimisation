# Module that defines the stochastic policies used by ants to select the paths associated with a
# given network.
# Module defining the functions used for updating the pheromone matrix.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
cimport numpy as np
import numpy as np
from .c_utils import rouletteWheel

INT8_DTYPE = np.int8
INT64_DTYPE = np.int64
DOUBLE_DTYPE = np.float64

ctypedef np.int8_t INT8_DTYPE_t
ctypedef np.int64_t INT64_DTYPE_t
ctypedef np.float64_t DOUBLE_DTYPE_t


cpdef np.ndarray[DOUBLE_DTYPE_t, ndim=1] stochasticAS(
        long int init_pos, np.ndarray[INT64_DTYPE_t, ndim=1] movements, np.ndarray[DOUBLE_DTYPE_t, ndim=2] H,
        np.ndarray[DOUBLE_DTYPE_t, ndim=2] P, double alpha):
    """
    Stochastic policy that determines the probability of an ant moving to each of the nodes based 
    on a trade-off between heuristic information and pheromone matrix information defined by:
    
        prob[i,j] = (P[i,j]^alpha * H[i, j]^beta) / (sum(P^alpha * H^beta))  
     
    From the probabilities and possible movements, applying a roulette wheel algorithm to the 
    computed probabilities it will return the position towards which the ant will move.
        
    Parameters
    ----------
    init_pos: long int
        Initial position of the ant in the connectivity graph to be traversed.
    
    movements: np.ndarray[np.int64_t, ndim=1]
        Possible movements the ant can take. For more information about how these movements are 
        computed see: antco/c_ntools.py/getValidPaths() 
    
    H np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Heuristic information.
        
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes) 
        Pheromone matrix with equal diagonal 0.
        
    alpha: float
        Parameter controlling the importance given to the pheromone matrix information.
        
        
    Returns
    -------
    :np.ndarray[DOUBLE_DTYPE_t, ndim=1]
        Probabilities associated with each move.
        
    Note
    ----
    It will be assumed that the exponentiation of the heuristic matrix to the beta parameter has 
    been pre-computed.
    """
    cdef int i, selected_index
    cdef int n_movements = movements.shape[0]
    cdef double phe_heu_product, sum_ = 0.0
    cdef np.ndarray[DOUBLE_DTYPE_t, ndim=1] probs = np.zeros(shape=n_movements, dtype=DOUBLE_DTYPE)
    cdef np.ndarray[DOUBLE_DTYPE_t, ndim=2] P_alpha_H_beta = np.multiply(np.power(P, alpha), H)

    # Calculate the sum of the denominator
    for i in range(n_movements):
        phe_heu_product = P_alpha_H_beta[init_pos, movements[i]]
        probs[i] = phe_heu_product
        sum_ += phe_heu_product

    # Calculate the values associated with each possible choice
    for i in range(n_movements):
        probs[i] /= (sum_ + 1e-06)

    return probs

