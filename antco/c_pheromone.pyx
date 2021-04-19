# Module defining the functions used for updating the pheromone matrix.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
cimport numpy as np
cimport cython

INT8_DTYPE = np.int8
INT64_DTYPE = np.int64
DOUBLE_DTYPE = np.float64

ctypedef np.int8_t INT8_DTYPE_t
ctypedef np.int64_t INT64_DTYPE_t
ctypedef np.float64_t DOUBLE_DTYPE_t


@cython.boundscheck(False)
cdef int argmax(np.ndarray[DOUBLE_DTYPE_t, ndim=1] array):
    """
    Function that returns the index of the highest number in the received array as long as it is 
    greater than -999,999.0.

    Parameters
    ----------
    array: np.ndarray[np.float64_t, ndim=1]
    
    Returns
    -------
    :int
        Index of the highest number. 
    """
    cdef int i, index, length = array.shape[0]
    cdef double max_value = -9_999_999.0

    for i in range(length):
        if array[i] > max_value:
            max_value = array[i]
            index = i

    return index


cpdef void updateUndAS(np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
                       np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, double rho, double weight):
    """
    Pheromone update strategy optimised for undirected graphs proposed in:

        "Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of
        cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
        26(1), 29-41"

    Pheromone levels are updated according to:

        P[i,j] = (1 - rho) * P[i,j] + sum_ants(P)

    where sum_ants(P) corresponds to the sum of the pheromone quantities left by the m ants in the
    edge (i,j) according to the following rule

    delta_P_ij = ant_scores[k] if ant k used edge (i,j) otherwise 0

    being q (amount of pheromone assigned to each ant) a constant and L[k] the tour built by ant k.

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes) 
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants) 
        Array indicating the scores associated with each of the ants.
    
    rho: double
        Parameter that reference the evaporation rate of pheromones.
    
    weight: double
        Parameter indicating the weight given to each of the ants.
        
    Notes
    -----
    This function performs a inplace modification in the pheromone matrix.
    """
    cdef int i, j, ant
    cdef int dim = P.shape[0], n_ants = paths.shape[0]
    cdef double delta, delta_P_ij, delta_pher

    # Since it is an undirected graph, it is only necessary to go through half of the elements
    for i in range(dim):
        for j in range(i+1, dim):
            delta_P_ij = 0.0
            for ant in range(n_ants):
                delta_pher = ant_scores[ant] * weight
                if paths[ant, i, j] == 1:
                    delta_P_ij += delta_pher
            delta = (1 - rho) * P[i, j] + delta_P_ij
            P[i, j], P[j, i] = delta, delta


cpdef void updateDirAS(np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
                       np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, double rho, double weight):
    """
    Pheromone update strategy optimised for directed graphs proposed in:

        "Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of
        cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
        26(1), 29-41"

    Pheromone levels are updated according to:

        P[i,j] = (1 - rho) * P[i,j] + sum_ants(P)

    where sum_ants(P) corresponds to the sum of the pheromone quantities left by the m ants in the
    edge (i,j) according to the following rule

    delta_P_ij = ant_scores[k] if ant k used edge (i,j) otherwise 0

    being q (amount of pheromone assigned to each ant) a constant and L[k] the tour built by ant k.

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes) 
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants) 
        Array indicating the scores associated with each of the ants.
    
    rho: double
        Parameter that reference the evaporation rate of pheromones.
    
    weight: double
        Parameter indicating the weight given to each of the ants.

    Notes
    -----
    This function performs a inplace modification in the pheromone matrix.
    """
    cdef int i, j, ant
    cdef int dim = P.shape[0], n_ants = paths.shape[0]
    cdef double delta_P_ij, delta_pher

    for i in range(dim):
        for j in range(dim):
            if not i == j:
                delta_P_ij = 0.0
                for ant in range(n_ants):
                    delta_pher = ant_scores[ant] * weight
                    if paths[ant, i, j] == 1:
                        delta_P_ij += delta_pher
                P[i, j] = (1 - rho) * P[i, j] + delta_P_ij


cpdef void updateDirEliteAS(np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
                            np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, double rho, int elite, double weight):
    """
    Pheromone update strategy optimised for directed graphs proposed in:

        "Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of
        cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
        26(1), 29-41"

    Pheromone levels are updated according to: (using only best ants)

        P[i,j] = (1 - rho) * P[i,j] + sum_ants(P)

    where sum_ants(P) corresponds to the sum of the pheromone quantities left by the m ants in the
    edge (i,j) according to the following rule

    delta_P_ij = ant_scores[k] if ant k used edge (i,j) otherwise 0

    being q (amount of pheromone assigned to each ant) a constant and L[k] the tour built by ant k.

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    elite: int
        Number of ants that will update the pheromone values.
    
    weight: double
        Parameter indicating the weight given to each of the ants.

    Notes
    -----
    This function performs a inplace modification in the pheromone matrix.
    """
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] best_indices = np.argsort(ant_scores)[::-1][:elite]
    cdef int ant_idx

    for ant_idx in range(elite):
        updateDirAS(paths[best_indices[ant_idx]:best_indices[ant_idx] + 1, :, :], P,
                           ant_scores[best_indices[ant_idx]:best_indices[ant_idx]+1], rho, weight)


cpdef void updateUndEliteAS(np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
                            np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, int elite, double weight):
    """
    Pheromone update strategy optimised for undirected graphs proposed in:

        "Dorigo, M., Maniezzo, V., & Colorni, A. (1996). Ant system: optimization by a colony of
        cooperating agents. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
        26(1), 29-41"

    Pheromone levels are updated according to: (using only best ants)

        P[i,j] = (1 - rho) * P[i,j] + sum_ants(P)

    where sum_ants(P) corresponds to the sum of the pheromone quantities left by the m ants in the
    edge (i,j) according to the following rule

    delta_P_ij = ant_scores[k] if ant k used edge (i,j) otherwise 0

    being q (amount of pheromone assigned to each ant) a constant and L[k] the tour built by ant k.

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    elite: int
        Number of ants that will update the pheromone values.
    
    weight: double
        Parameter indicating the weight given to each of the ants.

    Notes
    -----
    This function performs a inplace modification in the pheromone matrix.

    """
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] best_indices = np.argsort(ant_scores)[::-1][:elite]
    cdef int ant_idx

    for ant_idx in range(elite):
        updateUndAS(paths[best_indices[ant_idx]:best_indices[ant_idx] + 1, :, :], P,
                           ant_scores[best_indices[ant_idx]:best_indices[ant_idx]+1], rho, weight)


cpdef void updateDirMMAS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, tuple limits, double weight):
    """
    Pheromone update strategy optimised for directed graphs based on

        "Stützle, T., & Hoos, H. H. (2000). MAX–MIN ant system. Future generation computer systems,
        16(8), 889-914."

    In this updating strategy only the best ants update the pheromone values according to the
    following expression:

    P[i,j] = (1 - p) * P[i,j] + delta_P_ij_best

    where delta_P_ij_best corresponds to the ant associated with max(ant_scores).

    In this case the update of the pheromone values is carried out from the ant with the best
    associated score. In the original paper they suggest the use of the best overall or the best at
    the interaction level. Given the problems of convergence and less exploratory capacity isPresent
    in the use of the best global, in the isPresent implementation the best ant at iteration level
    will be used.

    In order to avoid that the pheromone values related to a certain path end up in a rapid
    convergence and consequently sub-optimal solutions, the pheromone values are limited between
    the limits [p_min, p_max]

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes) 
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants) 
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    limits: tuple
        Tuple of two elements with the limit values that pheromones can reach.
        
    weight: double
        Parameter indicating the weight given to each of the ants.
    """

    cdef int best_ant = argmax(ant_scores)
    cdef double lower_lim = limits[0], upper_lim = limits[1]
    cdef int dim = P.shape[0]
    cdef int i, j
    cdef double delta
    cdef double delta_pher = ant_scores[best_ant] * weight

    for i in range(dim):
        for j in range(dim):
            if not i == j:
                if paths[best_ant, i, j] == 1:
                    delta = (1 - rho) * P[i, j] + delta_pher
                else:
                    delta = (1 - rho) * P[i, j]

                if delta <= lower_lim:
                    P[i, j] = lower_lim
                elif delta >= upper_lim:
                    P[i, j] = upper_lim
                else:
                    P[i, j] = delta


cpdef void updateDirEliteMMAS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, tuple limits, int elite,
        double weight):
    """
    Pheromone update strategy optimised for directed graphs based on

        "Stützle, T., & Hoos, H. H. (2000). MAX–MIN ant system. Future generation computer systems,
        16(8), 889-914."

    In this updating strategy only the bag_size ants update the pheromone values according to the
    following expression:

    P[i,j] = (1 - p) * P[i,j] + delta_P_ij_best

    where delta_P_ij_best corresponds to the ant associated with max(ant_scores).

    In this case the update of the pheromone values is carried out from the ant with the best
    associated score. In the original paper they suggest the use of the best overall or the best at
    the interaction level. Given the problems of convergence and less exploratory capacity isPresent
    in the use of the best global, in the isPresent implementation the best ant at iteration level
    will be used.

    In order to avoid that the pheromone values related to a certain path end up in a rapid
    convergence and consequently sub-optimal solutions, the pheromone values are limited between
    the limits [p_min, p_max]

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    limits: tuple
        Tuple of two elements with the limit values that pheromones can reach.
    
    elite: int
        Number of ants that will update the pheromone values.
        
    weight: double
        Parameter indicating the weight given to each of the ants.
    """
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] best_indices = np.argsort(ant_scores)[::-1][:elite]
    cdef int ant_idx

    for ant_idx in range(elite):
        updateDirMMAS(
            paths[best_indices[ant_idx]:best_indices[ant_idx]+1, :, :], P,
            ant_scores[best_indices[ant_idx]:best_indices[ant_idx]+1], rho, limits, weight)


cpdef void updateUndMMAS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, tuple limits, double weight):
    """
    Pheromone update strategy optimised for undirected graphs based on

        "Stützle, T., & Hoos, H. H. (2000). MAX–MIN ant system. Future generation computer systems,
        16(8), 889-914."

    In this updating strategy only the best ants update the pheromone values according to the
    following expression:

    P[i,j] = (1 - p) * P[i,j] + delta_P_ij_best

    where delta_P_ij_best corresponds to the ant associated with max(ant_scores).

    In this case the update of the pheromone values is carried out from the ant with the best
    associated score. In the original paper they suggest the use of the best overall or the best at
    the interaction level. Given the problems of convergence and less exploratory capacity isPresent
    in the use of the best global, in the isPresent implementation the best ant at iteration level
    will be used.

    In order to avoid that the pheromone values related to a certain path end up in a rapid
    convergence and consequently sub-optimal solutions, the pheromone values are limited between
    the limits [p_min, p_max]

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes) 
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants) 
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    limits: tuple
        Tuple of two elements with the limit values that pheromones can reach.
    
    weight: double
        Parameter indicating the weight given to each of the ants.
    """

    cdef int best_ant = argmax(ant_scores)
    cdef double lower_lim = limits[0], upper_lim = limits[1]
    cdef int dim = P.shape[0]
    cdef int i, j
    cdef double delta
    cdef double delta_pher = ant_scores[best_ant] * weight

    for i in range(dim):
        for j in range(i+1, dim):
            if paths[best_ant, i, j] == 1:
                delta = (1 - rho) * P[i, j] + delta_pher
            else:
                delta = (1 - rho) * P[i, j]

            if delta <= lower_lim:
                P[i, j], P[j, i] = lower_lim, lower_lim
            elif delta >= upper_lim:
                P[i, j], P[j, i] = upper_lim, upper_lim
            else:
                P[i, j], P[j, i] = delta, delta


cpdef void updateUndEliteMMAS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, double rho, tuple limits, int elite,
        double weight):
    """
    Pheromone update strategy optimised for undirected graphs based on

        "Stützle, T., & Hoos, H. H. (2000). MAX–MIN ant system. Future generation computer systems,
        16(8), 889-914."

    In this updating strategy only the bag_size ants update the pheromone values according to the
    following expression:

    P[i,j] = (1 - p) * P[i,j] + delta_P_ij_best

    where delta_P_ij_best corresponds to the ant associated with max(ant_scores).

    In this case the update of the pheromone values is carried out from the ant with the best
    associated score. In the original paper they suggest the use of the best overall or the best at
    the interaction level. Given the problems of convergence and less exploratory capacity isPresent
    in the use of the best global, in the isPresent implementation the best ant at iteration level
    will be used.

    In order to avoid that the pheromone values related to a certain path end up in a rapid
    convergence and consequently sub-optimal solutions, the pheromone values are limited between
    the limits [p_min, p_max]

    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
    
    limits: tuple
        Tuple of two elements with the limit values that pheromones can reach.
    
    elite: int
        Number of ants that will update the pheromone values.
        
    weight: double
        Parameter indicating the weight given to each of the ants.
    """
    cdef np.ndarray[INT64_DTYPE_t, ndim=1] best_indices = np.argsort(ant_scores)[::-1][:elite]
    cdef int ant_idx

    for ant_idx in range(elite):
        updateUndMMAS(
            paths[best_indices[ant_idx]:best_indices[ant_idx]+1, :, :], P,
            ant_scores[best_indices[ant_idx]:best_indices[ant_idx]+1], rho, limits, weight)


cpdef void updateDirACS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, double weight):
    """ 
    Strategy for updating the pheromone matrix based on
    
        Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning
        approach to the traveling salesman problem. IEEE Transactions on evolutionary computation,
        1(1), 53-66.
        
    given by equation
                
        if edge (i,j) is in the path of the best ant:
        
            P[i,j] =  (1 - rho) * P[i,j] + rho * score * weight
        
        else:
        
            P[i,j] = P[i,j]
    
    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
        
    weight: double
        Parameter indicating the weight given to each of the ants.
        
    Notes
    -----
    Function optimised for directed graphs.
    """
    cdef int best_ant = argmax(ant_scores)
    cdef int dim = P.shape[0]
    cdef int i, j
    cdef double delta_pher = rho * ant_scores[best_ant] * weight
    cdef double rho_weight = (1 - rho)

    for i in range(dim):
        for j in range(dim):
            if not i == j:
                if paths[best_ant, i, j] == 1:
                    P[i, j] = rho_weight * P[i, j] + delta_pher



cpdef void updateUndACS(
        np.ndarray[INT8_DTYPE_t, ndim=3] paths, np.ndarray[DOUBLE_DTYPE_t, ndim=2] P,
        np.ndarray[DOUBLE_DTYPE_t, ndim=1] ant_scores, float rho, double weight):
    """ 
    Strategy for updating the pheromone matrix based on
    
        Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning
        approach to the traveling salesman problem. IEEE Transactions on evolutionary computation,
        1(1), 53-66.
        
    given by equation
                
        if edge (i,j) is in the path of the best ant:
        
            P[i,j] =  (1 - rho) * P[i,j] + rho * score * weight
        
        else:
        
            P[i,j] = P[i,j]
    
    Parameters
    ----------
    paths: np.ndarray[np.int8_t, ndim=3] of shape (n_ants, n_nodes, n_nodes)
        Array of integer numbers indicating the paths selected by each of the ants.
    
    P: np.ndarray[np.float64_t, ndim=2] (n_nodes, n_nodes)
        Pheromone matrix with equal diagonal 0.
    
    ant_scores: np.ndarray[np.float64_t, ndim=1] of shape (n_ants)
        Array indicating the scores associated with each of the ants.
    
    rho: float
        Parameter that reference the evaporation rate of pheromones.
        
    weight: double
        Parameter indicating the weight given to each of the ants.
        
    Notes
    -----
    Function optimised for undirected graphs.
    """
    cdef int best_ant = argmax(ant_scores)
    cdef int dim = P.shape[0]
    cdef int i, j
    cdef double increment
    cdef double delta_pher = rho * ant_scores[best_ant] * weight
    cdef double rho_weight = (1 - rho)

    for i in range(dim):
        for j in range(i+1, dim):
            if not i == j:
                if paths[best_ant, i, j] == 1:
                    increment = rho_weight * P[i, j] + delta_pher
                    P[i, j], P[j, i] = increment, increment
