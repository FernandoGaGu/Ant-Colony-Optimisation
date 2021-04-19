# Module with the central object that processes the data provided by the user and provides the
# basic functions to create ant-based algorithms.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import joblib
import sys
import warnings
from .ant import Ant, randomInit
from .optim import ObjectiveFunction
from .pheromone import updateAS, updateMMAS, updateACS, updateUndLocalPher, updateDirLocalPher
from .c_policy import stochasticAS
from .c_ntools import getValidPaths
from .c_utils import rouletteWheel
from .base import ScoreScaler

warnings.filterwarnings('error')

# Implemented pheromone update strategies
PHEROMONE_UPDATE_STRATEGIES = ['as', 'mmas', 'acs']


class ACO(object):
    """
    Basic object defining the parameters used by the ant-based optimisation algorithm. This object
    can be passed to some function from antco.algorithm module to execute the optimisation.

    Parameters
    ----------
    n_ants: int
        Number of ants used by the algorithm.

    graph: np.ndarray (nodes, nodes)
        Binary adjacency matrix that defines the structure of the graph to to be traversed.

    objective: antco.optim.ObjectiveFuntion subclass
        Instance of a subclass belonging to antco.optim.ObjectiveFunction that implements the
        evaluate() method in such a way that given an ant returns a scalar value.

    iterations: int
        Maximum number of iterations that the algorithm will carry out if the tolerance criterion
        has not been met before.

    heuristic: np.ndarray (nodes, nodes), dtype np.float64, default=np.ones(shape=(nodes, nodes))
        Heuristic information matrix. If no heuristic information matrix is provided, one with 
        all values equal to 1 will be created.

    **kwargs:
        evaporation: float, default=0.2
            Parameter that reference the evaporation rate of pheromones deposited on the edges of
            the graph.

        alpha: float, default=1.0
            The alpha parameter refers to the influence of pheromones when the ant makes a decision
            on the path through the walk being constructed.

        beta: float, default=1.0
            Analogous to the alpha parameter, the beta parameter refers to the importance given
            to the heuristic information received in the heuristic parameter.

        graph_type: str, default='directed'
            Parameter that indicates the representation of the graph to be explored in order to
            apply optimized algorithms for directed or undirected graphs. For undirected graphs
            use: 'undirected' or 'u'; for directed graphs use: 'directed' or 'd'.

        path_limits: tuple, default=(0, np.inf)
            Parameter that establishes the limits on the paths that ants can create in the graph.

        pheromone_init: float, default=1.0
            Pheromone initial value.

        fixed_position: bool, default=False
            Parameter indicating whether to place ants in a fixed position in the graph with no
            repeated initials positions in all the colony.

        pheromone_update: dict or str, default='AS'
            Strategy for updating pheromone values. The strategy can be provided as a string or as
            a dictionary. If it is specified as a string, no additional arguments can be included
            and they will be selected by default. In case it is provided as a dictionary additional
            parameters can be specified, in this situation the parameters can be provided as a
            dictionary where the key "strategy" will indicate the strategy to be used and the rest
            of the key-values the parameters associated to the selected strategy. The currently
            implemented strategies include:

                Ant System:

                    >>> colony = ACO(... pheromone_update='AS')

                    >>> colony = ACO(... pheromone_update={'strategy': 'AS'})

                    >>> colony = ACO(... pheromone_update={'strategy': 'AS', 'bag_size': 10,
                    >>>                                    'weight': 0.2})


                MAX-MIN Ant System:

                    >>> colony = ACO(... pheromone_update='MMAS')

                    >>> colony = ACO(... pheromone_update={'strategy': 'MMAS', 'limits': (0, 2)})

                    >>> colony = ACO(... pheromone_update={'strategy': 'MMAS', 'limits': (0, 2),
                    >>>                                    'bag_size': 10, 'weight': 0.2})

                    In this strategy, if no limits are provided these will be set to 0 and inf by
                    default.

                For more information on pheromone update methods use help(antco.undASupdate) or
                help(antco.undASupdate) for Ant System and help(antco.dirMMASupdate) or
                help(antco.undMMASupdate) for MAX-MIN Ant System.

        tol: int, default=np.inf
            This parameter corresponds to the maximum number of interactions without any improvement
            on the maximum value found before stopping the algorithm early. By default disabled.

        n_jobs: int, default=1
            Number of processes to run in parallel.

        precompute_heuristic: bool, default=True
            Indicates whether to exponentialise the values of the heuristic matrix to the beta
            parameter. If the value is set to False, the values will be exponentiated every time
            an ant performs a route (computationally more inefficient but necessary when the values
            of the heuristic matrix are not static). On the other hand, if the value is set to True,
            the values will be exponentiated only once at the beginning of the execution of the
            algorithm (computationally more efficient).

        Q: float, default None
            Parameter that determines the probability of selecting the next move deterministically 
            by selecting the move to the node that has the highest probability. By default this 
            parameter will not be considered.

        R: float, default None
            Parameter that determines the probability of selecting the next move randomly without 
            taking into account the computation of the pheromone matrix and heuristic information.
            By default  this parameter will not be considered.

        scaleScores: antco.base.ScoreScaler subclass, default None
            Subclass of antco.base.Scaler that allows scaling the scores to a certain range of
            values.

        seed: int, default=None
            Random seed.
    """

    def __init__(self, *, n_ants: int, graph: np.ndarray, objective: ObjectiveFunction, 
                 iterations: int, heuristic: np.ndarray = None, **kwargs):
        
        if heuristic is None:  # Heuristic information will have no influence on optimisation
            heuristic = np.ones(shape=graph.shape, dtype=np.int8)

        ACO.checkParameters({'n_ants': n_ants, 'graph': graph, 'heuristic': heuristic,
                             'objectiveFunction': objective, 'iterations': iterations}, kwargs)

        self._graph = np.array(graph, dtype=np.int8)
        self._H = heuristic.astype(np.float64)
        self._n_nodes = self._graph.shape[0]
        self._iterations = iterations
        self._objectiveFunction = objective
        self._updatePheromones = ACO._selectPheromoneUpdate(kwargs)
        self._pheromone_update_kw = ACO._selectPheromoneUpdateKW(kwargs)
        self._pheromone_update_kw['rho'] = kwargs.get('evaporation', 0.5)
        self._alpha = kwargs.get('alpha', 1.0)
        self._beta = kwargs.get('beta', 1.0)
        self._path_limits = kwargs.get('path_limits', (0, sys.maxsize))
        self._precompute_heuristic = kwargs.get('precompute_heuristic', True)
        self._P = np.zeros(shape=self._H.shape, dtype=np.float64)[:]
        self._P[:] = kwargs.get('pheromone_init', 1.0)
        self._pher_init = kwargs.get('pheromone_init', 1.0)
        np.fill_diagonal(self._P, 0)
        self._n_jobs = kwargs.get('n_jobs', 1)
        self._seed = kwargs.get('seed', None)
        self._tol = kwargs.get('tol', np.inf)
        self._fixed = kwargs.get('fixed_position', False)
        self._accessory_node = None
        self._Q = kwargs.get('Q', None)
        self._R = kwargs.get('R', None)
        self._scaleScores = kwargs.get('scaleScores', None)

        # Initialize ants
        self._ants = [
            Ant(l_min=self._path_limits[0], l_max=self._path_limits[1],
                graph_type=kwargs.get('graph_type', 'd')) for _ in range(n_ants)]

    def __repr__(self):
        return f'ACO(iterations={self._iterations}, pheromoneUpd={self._updatePheromones}, ' \
               f'alpha={self._alpha}, beta={self._beta}, path_limits={self._path_limits}, ' \
               f'tol={self._tol}, Q={self._Q}, R={self._R}, accessory_node={self._accessory_node}, ' \
               f'scaleScores={self._scaleScores}, n_jobs={self._n_jobs})'

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def checkParameters(args: dict, optional_args: dict):
        """ Function that checks the validity of the received arguments. """
        def _checkType(name: str, value: object, required: type):
            assert isinstance(value, required), 'Parameter: %s must be %s.' % (name, str(required))

        def _intType(name: str, value: int, min_val: int = None, max_val: int = None):
            _checkType(name, value, int)
            if min_val is not None:
                assert value > min_val, 'Parameter: %s must be grater than %d' % (name, min_val)
            if max_val is not None:
                assert value < max_val, 'Parameter: %s must be less than %d' % (name, max_val)

        def _floatType(name: str, value: float, min_val: float = None, max_val: float = None):
            _checkType(name, value, float)
            if min_val is not None:
                assert value > min_val, 'Parameter: %s must be grater than %.5f' % (name, min_val)
            if max_val is not None:
                assert value < max_val, 'Parameter: %s must be less than %.5f' % (name, max_val)

        # Check types
        _checkType('graph', args['graph'], np.ndarray)
        _checkType('heuristic', args['heuristic'], np.ndarray)
        _checkType('objectiveFunction', args['objectiveFunction'], ObjectiveFunction)
        _intType('n_ants', args['n_ants'], min_val=0)
        _intType('iterations', args['iterations'], min_val=0)

        # Check graph and heuristic dimensions (n_nodes, n_nodes)
        assert len(args['graph'].shape) == 2, \
            'Parameter graph must be a square matrix (n_nodes, n_nodes). Provided shape: {}'.\
                format(args['graph'].shape)
        assert len(args['heuristic'].shape) == 2, \
            'Parameter heuristic must be a square matrix (n_nodes, n_nodes). Provided shape: {}'.\
                format(args['heuristic'].shape)
        assert args['graph'].shape[0] == args['graph'].shape[1], \
            'Parameter graph must be a square matrix (n_nodes, n_nodes). Provided shape: {}'.\
                format(args['graph'].shape)
        assert args['heuristic'].shape[0] == args['heuristic'].shape[1], \
            'Parameter heuristic must be a square matrix (n_nodes, n_nodes). Provided shape: {}'.\
                format(args['heuristic'].shape)
        assert args['graph'].shape[0] == args['heuristic'].shape[0], \
            'The dimensions of the graph and heuristic parameters must match. Provided shapes: ' \
            'graph: %d; heuristic: %d' % (args['graph'].shape[0], args['heuristic'].shape[0])

        # Check optional arguments
        if optional_args.get('evaporation', None) is not None:
            _floatType('evaporation', optional_args['evaporation'], min_val=0.0)
        if optional_args.get('alpha', None) is not None:
            _floatType('alpha', optional_args['alpha'], min_val=0.0)
        if optional_args.get('beta', None) is not None:
            _floatType('beta', optional_args['beta'], min_val=0.0)
        if optional_args.get('fixed_position', None) is not None:
            _checkType('fixed_position', optional_args['fixed_position'], bool)
        if optional_args.get('Q', None) is not None:
            _floatType('Q', optional_args['Q'], min_val=0.0, max_val=1.0)
        if optional_args.get('R', None) is not None:
            _floatType('R', optional_args['R'], min_val=0.0, max_val=1.0)
        if optional_args.get('graph_type', None) is not None:
            assert (optional_args['graph_type'] == 'd') or (optional_args['graph_type'] == 'u') or \
                   (optional_args['graph_type'] == 'directed') or (optional_args['graph_type'] == 'undirected'), \
                'The accepted values for the graph_type parameter are: "directed" or "d" for directed ' \
                'graphs and "undirected" or "u" for undirected graphs.'
        if optional_args.get('path_limits', None) is not None:
            assert isinstance(optional_args['path_limits'], list) or \
                   isinstance(optional_args['path_limits'], tuple), 'Parameter: path_limits must be a tuple or a list'
            assert len(optional_args['path_limits']) == 2, \
                'The limits of the length of the path travelled by the ants must be defined between two integers.'
        if optional_args.get('pheromone_init', None) is not None:
            _floatType('pheromone_init', optional_args['pheromone_init'])
        if optional_args.get('tol', None) is not None:
            _intType('tol', optional_args['tol'], min_val=5)
        if optional_args.get('pheromone_update', None) is not None:
            assert isinstance(optional_args['pheromone_update'], str) or \
                   isinstance(optional_args['pheromone_update'], dict), \
                'The pheromone_update parameter must be provided as a string or a dictionary as ' \
                'specified in the documentation.'
            if isinstance(optional_args['pheromone_update'], str):
                assert optional_args['pheromone_update'].lower() in PHEROMONE_UPDATE_STRATEGIES, \
                    'Pheromone update strategy %s not recognised. Available strategies: %r' % \
                    (optional_args['pheromone_update'], PHEROMONE_UPDATE_STRATEGIES)
            else:
                if optional_args['pheromone_update'].get('bag_size', None) is not None:
                    _intType('pheromone_update "bag_size"', optional_args['pheromone_update']['bag_size'],
                              min_val=1, max_val=args['n_ants'])
                assert 'strategy' in optional_args['pheromone_update'], \
                    'Hormone update strategy not defined using the "strategy" key. Available strategies: %r' % \
                    PHEROMONE_UPDATE_STRATEGIES
                assert optional_args['pheromone_update']['strategy'].lower() in PHEROMONE_UPDATE_STRATEGIES, \
                    'Pheromone update strategy %s not recognised. Available strategies: %r' % \
                    (optional_args['pheromone_update']['strategy'], PHEROMONE_UPDATE_STRATEGIES)
        if optional_args.get('scaleScores', None) is not None:
            _checkType('scaleScores', optional_args['scaleScores'], ScoreScaler)
        if optional_args.get('seed', None) is not None:
            _intType('seed', optional_args['seed'], min_val=0)

    @staticmethod
    def _selectPheromoneUpdate(kwargs: dict):
        """ Method that selects the pheromone update strategy provided by the user. """
        unrecognised_strategy = \
            'Pheromone update strategy not recognized, please check the parameter ' \
            'pheromone_update. Available strategies: %r' % PHEROMONE_UPDATE_STRATEGIES

        # Default pheromone update strategy
        if kwargs.get('pheromone_update', None) is None:
            update_strategy_ = updateAS('D')
        elif isinstance(kwargs['pheromone_update'], str):
            if kwargs['pheromone_update'].lower() == 'as':
                update_strategy_ = updateAS('D')
            elif kwargs['pheromone_update'].lower() == 'mmas':
                update_strategy_ = updateMMAS('D')
            elif kwargs['pheromone_update'].lower() == 'acs':
                update_strategy_ = updateACS('D')
            else:
                assert False, unrecognised_strategy
        elif isinstance(kwargs['pheromone_update'], dict):
            if 'strategy' not in kwargs['pheromone_update']:
                assert False, 'The key "strategy" in charge of specifying the update strategy is ' \
                              'not isPresent in the pheromone_update parameter.'
            if kwargs['pheromone_update']['strategy'].lower() == 'as':
                update_strategy_ = updateAS(kwargs.get('graph_type', 'D'),
                                            kwargs['pheromone_update'].get('elite', False))
            elif kwargs['pheromone_update']['strategy'].lower() == 'mmas':
                update_strategy_ = updateMMAS(kwargs.get('graph_type', 'D'),
                                              kwargs['pheromone_update'].get('elite', False))
            elif kwargs['pheromone_update']['strategy'].lower() == 'acs':
                update_strategy_ = updateACS(kwargs.get('graph_type', 'D'))
            else:
                assert False, unrecognised_strategy
        else:
            assert False, 'Unrecognised error in colony.ACO._selectPheromoneUpdate()'

        return update_strategy_

    @staticmethod
    def _selectPheromoneUpdateKW(kwargs: dict):
        """ Method that selects the optional arguments associated with the pheromone update and
        returns them as a dictionary. """
        if kwargs.get('pheromone_update', None) is None:
            return {'weight': 1.0}

        if isinstance(kwargs.get('pheromone_update', None), str):
            if kwargs['pheromone_update'].lower() == 'as':
                return {'weight': 1.0}
            elif kwargs['pheromone_update'].lower() == 'mmas':
                return {'limits': (0, 1), 'weight': 1.0}
            elif kwargs['pheromone_update'].lower() == 'acs':
                return {'decay': 0.1, 'weight': 1.0}

        elif isinstance(kwargs.get('pheromone_update', None), dict):
            if kwargs['pheromone_update']['strategy'].lower() == 'as':
                if kwargs['pheromone_update'].get('elite', None) is not None:
                    return {'elite': kwargs['pheromone_update']['elite'],
                            'weight': kwargs['pheromone_update'].get('weight', 1.0)}
                return {'weight': 1.0}
            elif kwargs['pheromone_update']['strategy'].lower() == 'mmas':
                if kwargs['pheromone_update'].get('elite', None) is not None:
                    return {'limits': tuple(kwargs['pheromone_update'].get('limits', (0, 1))),
                            'elite': kwargs['pheromone_update']['elite'],
                            'weight': kwargs['pheromone_update'].get('weight', 1.0)}
                return {'limits': tuple(kwargs['pheromone_update'].get('limits', (0, 1))),
                        'weight': kwargs['pheromone_update'].get('weight', 1.0)}
            elif kwargs['pheromone_update']['strategy'].lower() == 'acs':
                return {'decay': kwargs['pheromone_update'].get('decay', 0.1),
                        'weight': kwargs['pheromone_update'].get('weight', 1.0)}
        else:
            assert False, 'Unrecognised pheromone_update in antco.colony.ACO._selectPheromoneUpdateKW()'

    @property
    def seed(self) -> int:
        """ Returns the selected random seed. """
        return self._seed

    @property
    def graph(self) -> np.ndarray:
        """ Returns the graph defined by an adjacency matrix. """
        return self._graph

    @graph.setter
    def graph(self, input_val: tuple):
        """ Select a new graph matrix, to prevent accidental changes this must be received as a
        tuple where the first element will be the new graph matrix and the second a boolean true
        value. """
        assert input_val[1], 'Incorrect graph assignation protocol.'

        self._graph = input_val[0]

    @property
    def iterations(self) -> int:
        """ Returns the number of iterations to be performed. """
        return self._iterations

    @property
    def n_jobs(self) -> int:
        """ Returns the number of processors used for the execution of the algorithm. """
        return self._n_jobs

    @property
    def ants(self) -> list:
        """ Returns the list of ants used by the algorithm. """
        return self._ants

    @property
    def heuristic(self) -> np.ndarray:
        """ Returns the heuristic matrix. """
        return self._H

    @heuristic.setter
    def heuristic(self, input_val: tuple):
        """ Select a new heuristic matrix, to prevent accidental changes this must be received as a
        tuple where the first element will be the new heuristic matrix and the second a boolean
        true value. """
        assert input_val[1], 'Incorrect heuristic assignation protocol.'

        self._H = input_val[0]

    @property
    def precompute_heuristic(self):
        return self._precompute_heuristic

    @property
    def pheromones(self) -> np.ndarray:
        """ Returns the initialized pheromone matrix. """
        return self._P

    @pheromones.setter
    def pheromones(self, input_val: tuple):
        """ Select a new pheromone matrix, to prevent accidental changes this must be received as
        a tuple where the first element will be the new pheromone matrix and the second a boolean
        true value. """
        assert input_val[1], 'Incorrect pheromone assignation protocol.'

        self._P = input_val[0]

    @property
    def alpha(self) -> float:
        """ Returns the alpha parameter. """
        return self._alpha

    @property
    def beta(self) -> float:
        """ Returns the beta parameter. """
        return self._beta

    @property
    def pheromone_update_kw(self) -> dict:
        """ Returns the optional parameters used by the pheromone update strategy. """
        return self._pheromone_update_kw

    @property
    def updatePheromones(self) -> callable:
        """ Returns the pheromone update strategy. """
        return self._updatePheromones

    @property
    def pher_init_val(self) -> float:
        """ Returns the pheromone initial value. """
        return self._pher_init

    @property
    def objectiveFunction(self) -> ObjectiveFunction:
        """ Returns the objectiveFunction function to be maximized. """
        return self._objectiveFunction

    @property
    def tol(self) -> float:
        """ Returns the tolerance criteria for early stopping. """
        return self._tol

    @property
    def fixed_positions(self) -> bool:
        """ Returns a boolean value indicating whether to place the ants in fixed positions or
        not."""
        return self._fixed

    @property
    def Q(self) -> float or None:
        """ Returns the probability of selecting the next move deterministically by selecting the 
        move to the node that has the highest probability."""
        return self._Q

    @property
    def R(self) -> float or None:
        """ Returns the probability of selecting the next move randomly without taking into account 
        the computation of the pheromone matrix and heuristics """
        return self._R

    @property
    def scaleScores(self) -> ScoreScaler:
        """ Specifies whether to scale scores to the range 0-1 using the Min-Max algorithm. """
        return self._scaleScores
    
    @property
    def accessory_node(self) -> int or None:
        """ Returns the position of the accessory node (for more information see antco.preproc
        module). """
        return self._accessory_node
    
    @accessory_node.setter
    def accessory_node(self, node: int):
        """ Select the accessory node. """
        assert isinstance(node, int), 'Accessory node must be an integer.'
        assert node == self._graph.shape[0], 'Accessory node out of graph.'

        self._accessory_node = node


def getRandomWalk(initial_position: int, current_path: np.ndarray, adjacency_matrix: np.ndarray,
                  heuristic: np.ndarray, pheromone: np.ndarray, alpha: float, max_lim: int,
                  Q: float or None, R: float or None) -> np.ndarray:
    """
    Function that given an array indicating the nodes traversed (path traversed by an ant), a
    binary adjacency matrix indicating the structure of the graph to be traversed and the
    parameters that regulate the stochastic choices that the ants will make when choosing their
    movements (alpha and beta parameters that regulates the influence of the pheromone and
    heuristic values on the decisions taken by the ants) returns a binary adjacency matrix
    indicating the path traversed by the ant.

    Parameters
    ----------
    initial_position: int
        Integer indicating the initial position of the ant.

    current_path: np.ndarray (nodes), dtype=np.int8
        Array with nodes visited by the ant. The current_path argument must include the initial
        position of the ant.

    adjacency_matrix: np.ndarray (nodes, nodes), dtype=np.int8
        Binary adjacency matrix defining the structure of the graph to be traversed.

    heuristic: np.ndarray (nodes, nodes), dtype=np.float64
        Heuristic information matrix used by the stochastic ant policy to decide the ant's
        movements.

    pheromone: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone information matrix used by the stochastic ant policy to decide the ant's
        movements. The parameters of this matrix will be updated throughout the interactions of
        the algorithm.

    alpha: float
        Parameter that reference the influence of pheromones when the ant makes a decision on the
        path through the walk being constructed.

    max_lim: int
        Maximum path length.

    Q: float, default=None
        Parameter that determines the probability of selecting the next move deterministically 
        by selecting the move to the node that has the highest probability. By default this 
        parameter will not be considered.

    R: float, default=None
        Parameter that determines the probability of selecting the next move randomly without 
        taking into account the computation of the pheromone matrix and heuristics. By default 
        this parameter will not be considered.


    Returns
    -------
    :np.ndarray (nodes)
        Array with the nodes visited by the ant arranged in the order in which they have been
        visited.
    """
    movements = getValidPaths(initial_position, current_path, adjacency_matrix)
    n_partial_solutions = 1

    # Add partial solutions to the current path as long as possible.
    while len(movements) > 0 and n_partial_solutions < max_lim:
    
        if len(movements) == 1:
            mov = movements[0]
        
        elif R is not None and np.random.uniform() < R:   # Random selection of the move
            mov = np.random.choice(movements)
            
        elif Q is not None and np.random.uniform() < Q:    # Deterministic selection of the move
            probs = stochasticAS(
                initial_position, np.array(movements), heuristic, pheromone, alpha)        
            mov = movements[np.argmax(probs)]
            
        else:  # Stochastic selection of the next move            
            probs = stochasticAS(
                initial_position, np.array(movements), heuristic, pheromone, alpha)

            mov = movements[rouletteWheel(probs)]

        current_path = np.append(current_path, mov)
        movements = getValidPaths(mov, current_path, adjacency_matrix)
        initial_position = mov
        n_partial_solutions += 1

    return current_path


def step(ant: Ant, adjacency_matrix: np.ndarray, heuristic: np.ndarray, pheromone: np.ndarray,
         alpha: float, beta: float, exp_heuristic: bool = True, Q: float = None, 
         R: float = None) -> Ant:
    """
    Basic step function which ensures that an ant makes a path around the graph. The steps carried
    out by this function include:

        1. Initial positioning of the ant in the graph to be explored.
        2. Calling the getRandomWalk function so that the ant traverses the network.
        3. Processing of the path returned by the function.


    Parameters
    ----------
    ant: Ant
        Ant. Important note: for parallelization reasons this function will not modify the internal
        state of the received ant, it will returns a new ant.

    adjacency_matrix: np.ndarray (nodes, nodes), dtype=np.int8
        Binary adjacency matrix that defines the structure of the graph to be covered.

    heuristic: np.ndarray (nodes, nodes), dtype=np.float64
        Heuristic information matrix used by stochastic politics to select the nodes towards which
        the ant will move.

    pheromone: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone information matrix used by stochastic politics to select the edges towards which
        the ant will move. This matrix will be optimised over successive iterations of the algorithm.

    alpha: float
        Parameter that reference the influence of pheromones when the ant makes a decision on the
        path through the walk being constructed.

    beta: float
        Analogous to the alpha parameter, the beta parameter reference the importance given
        to the heuristic information.

    exp_heuristic: bool, default=False
        Boolean value indicating whether to exponentiate the heuristic matrix to beta or not. By
        default it will not be exponentiated. It will be assumed that it has been previously
        exponentiated. Set this parameter to True for problems where the values of the heuristic
        matrix are changing throughout the interactions of the algorithm.

    Q: float, default=None
        Parameter that determines the probability of selecting the next move deterministically 
        by selecting the move to the node that has the highest probability. By default this 
        parameter will not be considered.

    R: float, default=None
        Parameter that determines the probability of selecting the next move randomly without 
        taking into account the computation of the pheromone matrix and heuristics. By default 
        this parameter will not be considered.

    Returns
    -------
    :Ant
        Returns the ant that has travelled along the network.
    """
    if exp_heuristic:
        heuristic = np.power(heuristic, beta)

    new_ant = Ant(l_min=ant.min_length, l_max=ant.max_length, graph_type=ant.representation,
                  check_params=False)

    new_ant.initAdjMatrix(n_nodes=adjacency_matrix.shape[0])

    if ant.initial_position is None:  # Random initial position
        initial_position = randomInit(adjacency_matrix)
    else:
        initial_position = ant.initial_position

    new_ant.setInitialPosition(initial_position)

    ant_path = getRandomWalk(
        new_ant.initial_position, new_ant.visited_nodes, adjacency_matrix, heuristic, pheromone,
        alpha, new_ant.max_length, Q, R)

    new_ant.visited_nodes = ant_path

    return new_ant


def generatePaths(ants: list, graph: np.ndarray, H: np.ndarray, P: np.ndarray, alpha: float,
                  beta: float, Q: float, R: float, n_jobs: int, exp_heuristic: bool = True) -> list:
    """
    Function that performs the exploration of the network according to the values of the heuristic
    and pheromone matrix.

    Parameters
    ----------
    ants: list
        List of ant instances.

    graph: np.ndarray (nodes, nodes), dtype=np.int8
        Graph to be explored.

    H: np.ndarray (nodes, nodes), dtype=np.float64
        Heuristic information.

    P: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone information.

    alpha: float
        The alpha parameter reference the influence of pheromones when the ant makes a decision
        on the path through the walk being constructed.

    beta: float
        Analogous to the alpha parameter, the beta parameter reference the importance given
        to the heuristic information received in H.

    Q: float, default=None
        Parameter that determines the probability of selecting the next move deterministically
        by selecting the move to the node that has the highest probability. By default this
        parameter will not be considered.

    R: float, default=None
        Parameter that determines the probability of selecting the next move randomly without
        taking into account the computation of the pheromone matrix and heuristics. By default
        this parameter will not be considered.

    n_jobs: int
        Number of processes to run in parallel.

    exp_heuristic: bool, default=True
        Parameter indicating whether to exponentiate the heuristic matrix to the beta value. By
        default it will not be assumed that the exponentiation has been precomputed.

    Returns
    -------
    :list
        List of ant instances that have traversed the graph to be explored.
    """
    if n_jobs == 1:
        ants = [
            step(ant, graph, H, P, alpha, beta, exp_heuristic=exp_heuristic, Q=Q, R=R)
            for ant in ants]
    else:
        ants = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(step)(
                ant, graph, H, P, alpha, beta, exp_heuristic=exp_heuristic, Q=Q, R=R)
            for ant in ants)

    return ants


def generatePathsACS(ants: list, graph: np.ndarray, H: np.ndarray, P: np.ndarray, alpha: float,
                     beta: float, decay: float, pher_init: float, Q: float,
                     exp_heuristic: bool = True) -> list:
    """
    Function that performs the exploration of the graph using the Ant Colony System strategy
    proposed in

        Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning
        approach to the traveling salesman problem. IEEE Transactions on evolutionary computation,
        1(1), 53-66.

    Parameters
    ----------
    ants: list
        List of ant instances.

    graph: np.ndarray (nodes, nodes), dtype=np.int8
        Graph to be explored.

    H: np.ndarray (nodes, nodes), dtype=np.float64
        Heuristic information.

    P: np.ndarray (nodes, nodes), dtype=np.float64
        Pheromone information.

    alpha: float
        The alpha parameter reference the influence of pheromones when the ant makes a decision
        on the path through the walk being constructed.

    beta: float
        Analogous to the alpha parameter, the beta parameter reference the importance given
        to the heuristic information received in H.

    decay: float
        Decay to be applied during the local update of the pheromone matrix values after an ant
        has made the tour. This parameter is used in the local pheromone update given by equation

            P[i,j] = (1 - decay) * P[i,j] + decay * pher_init

    pher_init: float
        Parameter involved in the local update of the pheromone matrix values according to the
        equation

            P[i,j] = (1 - decay) * P[i,j] + decay * pher_init

    Q: float, default=None
        Parameter that determines the probability of selecting the next move deterministically
        by selecting the move to the node that has the highest probability. By default this
        parameter will not be considered.

    exp_heuristic: bool, default=True
        Parameter indicating whether to exponentiate the heuristic matrix to the beta value. By
        default it will not be assumed that the exponentiation has been precomputed.

    Returns
    -------
    :list
        List of ant instances that have traversed the graph to be explored.

    """
    if exp_heuristic:
        H_beta = np.power(H, beta)
    else:
        H_beta = H

    new_ants = []
    for ant in ants:
        new_ant = Ant(l_min=ant.min_length, l_max=ant.max_length, graph_type=ant.representation,
                      check_params=False)

        new_ant.initAdjMatrix(n_nodes=graph.shape[0])
        init_pos = randomInit(graph) if new_ant.initial_position is None else new_ant.initial_position
        new_ant.setInitialPosition(init_pos)

        # Generate random walk
        new_ant.visited_nodes = getRandomWalk(
            initial_position=new_ant.initial_position, current_path=new_ant.visited_nodes,
            adjacency_matrix=graph, heuristic=H_beta, pheromone=P, alpha=alpha,
            max_lim=new_ant.max_length, Q=Q, R=None)

        # Local pheromone update
        if new_ant.representation == 'u':
            updateUndLocalPher(ant=new_ant, P=P, decay=decay, init_val=pher_init)
        else:
            updateDirLocalPher(ant=new_ant, P=P, decay=decay, init_val=pher_init)

        new_ants.append(new_ant)

    return new_ants


def evaluateAnts(ants: list, objectiveFunction: ObjectiveFunction, parallel_evaluation: bool,
                 n_jobs: int = 1) -> np.ndarray:
    """
    Function that performs the evaluation of the paths traversed by the ants using the defined cost
    function.

    Parameters
    ----------
    ants: list
        List of ant instances.

    objectiveFunction: antco.optim.ObjectiveFunction
        Subclass of antco.optim.ObjectiveFunction defined by the user. This function will be
        maximized.

    parallel_evaluation: bool
        Parameter indicating whether to carry out the evaluation in parallel.

    n_jobs: int
        If parallel_evaluation is True, number of processes to run in parallel.

    Returns
    -------
    :np.ndarray (len(ants)), dtype=np.float64
        Scores associated with the ants.
    """
    if parallel_evaluation:
        ant_scores = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(objectiveFunction)((ant, idx)) for idx, ant in enumerate(ants))
        # HACK: Parallel execution does not return indexes in order
        for score, idx in ant_scores:
            ant_scores[idx] = score
    else:
        ant_scores = np.array([
            objectiveFunction((ant, idx)) for idx, ant in enumerate(ants)], dtype=np.float64)[:, 0]

    return ant_scores

