# Module that defines the data structure representing the virtual ants of the algorithm and the
# functions applicable to this data structure.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from .c_ntools import toDirAdjList, toUndAdjList, toDirAdjMatrix, toUndAdjMatrix


class Ant(object):
    """
    Object representing an ant.

    Parameters
    ----------
    l_min: int or float
        Minimum length of the path traversed by the ant for the solution found to be valid (i.e.
        is_valid to return True).

    l_max: int or float
        Maximum length of the path traversed by the ant for the solution found to be valid (i.e.
        is_valid to return True).

    graph_type: str, default='directed'
        Type of graph representation to be traversed by the ant. This parameter is necessary
        because depending on the type of the graph, different optimised algorithms will be used.
        Possible values:
            - 'd' or 'directed' for directed graphs.
            - 'u' or 'undirected' for undirected graphs.

    check_params: bool, default=True
        Parameter indicating whether to check the parameters received as arguments when creating
        an Ant instance. By default this value shall be set to True.

    Example
    --------
    >>> from antco import Ant, getValidPaths

    Create an ant that is going to operate on a directed network which, in order to be valid, must
    travel along paths containing from 1 to 10 edges.

    >>> ant_u = antco.Ant(1, 10, 'u')  # Ant for undirected graphs
    >>> ant = antco.Ant(1, 10, 'd')    # Ant for directed graphs
    >>> ant.is_valid
    In [0]: False
    >>> ant.max_length
    In [1]: 10
    >>> ant.min_length
    In [2]: 1
    >>> ant.representation
    In [3]: 'd'

    Initialise an empty binary adjacency matrix (for a graph of 10 nodes) to save the paths travelled.

    >>> ant.initAdjMatrix(n_nodes=10)

    Select the starting position of the ant.

    >>> ant.setInitialPosition(4)
    >>> ant.initial_position
    In [4]: 4

    Select initial position to a random (valid) position in a graph

    >>> adj_matrix = ...  # Binary adjacency matrix indicating the graph connections
    >>> ant.setInitialPosition(antco.randomInit(ant))
    >>> ant.initial_position
    In [5]: 2

    Distribute several ants along a network avoiding putting more than two ants in the same
    position as long as there are valid empty positions in the network.

    >>> ants = [antco.Ant(1, 10, 'u') for _ in range(10)]
    >>> # Initialise a densely connected undirected graph.
    >>> adj_matrix = np.ones(shape=(10, 10))
    >>> np.fill_diagonal(adj_matrix, 0)   # Fill diagonal with 0s
    >>> antco.fixedPositions(ants, adj_matrix)
    >>> # Print ant positions
    >>> ' - '.join(str(ant.initial_position) for ant in ants)
    In [6]: '8 - 2 - 3 - 0 - 5 - 7 - 1 - 6 - 4 - 9'

    Eliminate the initial positions of the ants.
    >>> antco.deleteInitialPosition(ants)
    >>> ' - '.join(str(ant.initial_position) for ant in ants)
    In [7]: 'None - None - None - None - None - None - None - None - None - None'

    Once the ant has traversed the graph (use help(antco.step) or see antco.step for more
    information) the list of nodes visited by the ant can be accessed using the attribute
    'visited_nodes':

    >>> ant.getVisitedNodes
    In [8]: array([3, 0, 0, 5, 6])
    >>> adj_matrix = ant.adj_matrix
    In [9]:
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=int8)

    Creation of an ant from the values of the called ant.
    >>> new_ant = ant.new(initial_position=3)
    >>> new_ant.min_length
    In [10]: 1
    >>> new_ant.max_length
    In [11]: 10
    >>> new_ant.getVisitedNodes
    In [12]: array([])
    """
    def __init__(self, l_min: int or float, l_max: int or float, graph_type: str = 'directed',
                 check_params: bool = True):
        if check_params:
            assert (isinstance(l_min, int) or isinstance(l_min, float)) and \
                   (isinstance(l_max, int) or isinstance(l_max, float)), \
                'l_min and l_max must be integers or floats.'
            assert l_min < l_max, 'The lower limit cannot be equal to or less than the upper limit.'
            assert l_min >= 0, 'The lower limit cannot be less than 0.'
            assert isinstance(graph_type, str), 'graph_type must be a string.'

            graph_type_ = graph_type.lower()
            assert (graph_type_ == 'd') or (graph_type_ == 'u') or (graph_type_ == 'directed') or \
                   (graph_type_ == 'undirected'), \
                'The accepted values for the graph_type parameter are: "directed" or "d" for directed ' \
                'graphs and "undirected" or "u" for undirected graphs.'

        graph_type_ = graph_type.lower()

        # Select the mapping functions between adjacency lists and adjacency matrices
        if graph_type_ == 'd' or graph_type_ == 'directed':
            self._toAdjList = toDirAdjList
            self._toAdjMatrix = toDirAdjMatrix
            self._graph_type = 'd'
        else:
            self._toAdjList = toUndAdjList
            self._graph_type = 'u'
            self._toAdjMatrix = toUndAdjMatrix

        self._path_limits = (l_min, l_max)
        self._adjacency_list = None
        self._adjacency_matrix = None
        self._initial_position = None
        self._mapped = False
        self._positioned = False

    def __repr__(self):
        return f'Ant(path_limits={self._path_limits}, graph_type={self._graph_type})'

    def __str__(self):
        return self.__repr__()

    def initAdjMatrix(self, n_nodes: int):
        """
        Method that creates an empty binary adjacency matrix of the specified number of nodes.

        Parameters
        ----------
        n_nodes: int
            Number of nodes of the adjacency matrix.
        """
        self._adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes), dtype=np.int8)
        np.fill_diagonal(self._adjacency_matrix, 1)
        self._mapped = False

    def setInitialPosition(self, position: int or None):
        """
        Method for selecting the initial position of the ant.

        Parameters
        ----------
        position: int
            Initial position.

        Notes
        -----
        For efficiency reasons the verification of the received values is disabled.
        """
        self._initial_position = position
        self._mapped = False
        self._positioned = True

    @property
    def initial_position(self) -> int:
        """ Method that returns the initial position of the ant. """
        return self._initial_position

    @property
    def visited_nodes(self) -> np.ndarray:
        """
        Returns an array with the visited nodes (the indices of the visited nodes, i.e. a list with
        the partial solutions).

        Returns
        -------
        :np.ndarray
            Array with the nodes visited by the ant.

        Notes
        -----
        Note the partial solutions found by an ant are internally represented as adjacent matrices.
        Calling this function transforms the adjacency matrix into an array. As long as the
        adjacency matrix is not modified the following calls to the function will use the previously
        computed array without the need to transform the adjacency matrix back into an array.
        """
        if self._adjacency_matrix is None or not self._positioned:
            return np.array([], dtype=np.int64)

        if self._mapped:
            return self._adjacency_list

        self._adjacency_list = self._toAdjList(self._initial_position, self._adjacency_matrix)
        self._mapped = True

        return self._adjacency_list

    @visited_nodes.setter
    def visited_nodes(self, adj_list):
        """ Method for assigning an adjacency list to an ant.  """
        assert len(adj_list) > 1, 'The adjacency list must have at least two nodes.'
        self.setInitialPosition(adj_list[0])
        self.adj_matrix = self._toAdjMatrix(np.array(adj_list, np.int64), self._adjacency_matrix.shape[0])

    @property
    def adj_matrix(self) -> np.ndarray or None:
        """
        Method that returns the binary adjacency matrix representing the path chosen by the ant. If
        the ant has no path the method will return None.

        Returns
        -------
        :np.ndarray (nodes, nodes) or None
             Binary adjacency matrix or None.
        """
        return self._adjacency_matrix

    @adj_matrix.setter
    def adj_matrix(self, new_adj_matrix: np.ndarray):
        """
        Setter that allows the selection of a new binary adjacency matrix for a given ant.

        Parameters
        ----------
        new_adj_matrix: np.ndarray (nodes, nodes), dtype=np.int8
            New integer square adjacency matrix where the positions with value 1 indicate the
            connection between the nodes indicated by the row and column indices.

        Notes
        -----
        For efficiency reasons this method will only check that the matrix received is square. IT
        WILL NOT CHECK THE TYPE OF DATA IN THE RECEIVED MATRIX.
        """
        assert len(new_adj_matrix.shape) == 2 and new_adj_matrix.shape[0] == new_adj_matrix.shape[1], \
            'The dimensions of the adjacency matrix received are not consistent, it must be a square ' \
            'matrix of integer type.'

        self._mapped = False
        self._adjacency_matrix = new_adj_matrix

    @property
    def max_length(self) -> int:
        """
        Method that informs about the maximum size allowed for the path taken by the ant.

        Returns
        -------
        :int
            Maximum path length.
        """
        return self._path_limits[1]

    @property
    def min_length(self) -> int:
        """
        Method that informs about the minimum size allowed for the path taken by the ant.

        Returns
        -------
        :int
            Minimum path length.
        """
        return self._path_limits[0]

    @property
    def is_valid(self) -> bool:
        """
        Method that informs about the minimum size restrictions allowed for a partial solution.

        Returns
        -------
        :bool
            True if the number of elements in the partial solution is within the permitted limits,
            otherwise False.
        """
        length = len(np.where(self._adjacency_matrix == 1)[0]) if self._graph_type == 'd' else \
                 len(np.where(self._adjacency_matrix == 1)[0]) // 2

        if self._path_limits[0] <= length <= self._path_limits[1]:
            return True

        return False

    @property
    def representation(self) -> str:
        """
        Returns the graph representation type that is to be walked through.

        Returns
        -------
        :str
            Representation: 'd' for directed graphs and 'u' for undirected.
        """
        return self._graph_type

    def new(self):
        """
        Method to construct an ant while preserving the l_min, l_max and graph_type values of the
        calling instance. This method will automatically handle the call to initAdjMatrix().

        Returns
        -------
        :Ant
            Instance of the Ant class with parameters l_min, l_max and graph_type equal to those of
            the calling instance.
        """
        new_ant = Ant(l_min=self.min_length, l_max=self.max_length, graph_type=self.representation)
        new_ant.initAdjMatrix(n_nodes=self.adj_matrix.shape[0])

        return new_ant


def randomInit(adjacency_matrix: np.ndarray) -> int:
    """
    Method that randomly selects a position within the binary adjacency matrix.

    Parameters
    ----------
    adjacency_matrix: np.ndarray of shape (n_nodes, n_nodes), dtype=np.int8
        Binary adjacency matrix that defines the structure of the network and whose diagonal
        should be 0.

    Returns
    -------
    :int
        Integer indicating the initial position of the ant on the graph selected at random.
    """
    available_indices = np.unique(np.where(adjacency_matrix == 1))

    return np.random.choice(available_indices)


def fixedPositions(ants: list, adjacency_matrix: np.ndarray) -> None:
    """
    Function that places ants in fixed positions at different positions in the adjacency matrix by
    distributing them uniformly across the graph.

    Parameters
    ----------
    ants: list
        List with ants that will be placed in fixed positions, their internal state will be modified.

    adjacency_matrix: np.ndarray of shape (n_nodes, n_nodes), dtype=np.int8
        Binary adjacency matrix that defines the structure of the network and whose diagonal should
        be 0.
    """
    available_positions = np.unique(np.array(np.where(adjacency_matrix == 1)))
    for ant in ants:
        i = np.random.randint(len(available_positions))
        ant.setInitialPosition(available_positions[i])
        # Eliminate indices where ants are positioned
        available_positions = np.delete(available_positions, i)
        # If all the indices are already filled with ants, start placing more ants on the visited indices
        if len(available_positions) == 0:
            available_positions = np.unique(np.array(np.where(adjacency_matrix == 1)))


def deleteInitialPosition(ants: list) -> None:
    """
    Method that selects the initial position of the list of ants to None.

    Parameters
    ----------
    ants: list
        List with ants that will be placed in fixed positions, their internal state will be modified.
    """
    for ant in ants:
        ant.setInitialPosition(None)
