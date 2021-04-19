# Module containing the necessary elements to monitor the evolution of the algorithms.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
from collections import defaultdict
from .c_metrics import getBranchingFactor
from .ant import Ant


class Report(object):
    """
    Object used to monitor the performance of a given algorithm.

    Methods
    -------
    computeMetrics(iteration: int, pheromones: np.ndarray, adj_matrix: np.ndarray, name: str)
    get(keyword: str): dict
    save(iteration: int, **kwargs)
    updateBest(iteration: int, score: float, ant: antco.Ant)


    Attributes
    ----------
    values: dict
    best_solution: dict

    Example
    -------
    >>> import antco
    >>>
    >>> report = antco.report.Report()

    To save parameters:

    >>> report.save(1, cost=1.0, performance=3.0)
    >>> report.save(2, cost=0.5, performance=4.0)
    >>> report.save(3, cost=0.1, performance=5.0)

    To retrieve the data:

    >>> report.get('cost')
    In[0]: {1: 1.0, 2: 0.5, 3: 0.1}

    >>> report.values
    In[1]: defaultdict(dict,
                      {1: {'cost': 1.0, 'performance': 3.0},
                       2: {'cost': 0.5, 'performance': 4.0},
                       3: {'cost': 0.1, 'performance': 5.0}})

    With these objects it is possible to monitor other aspects of the convergence of the algorithm,
    for example the lambda-branching factor

    >>> report = Report({
    >>>             'getBranchingFactor': [{'lambda_values': [0.05, 0.1]}]
    >>>                 })

    In the above example the lambda-branching factor will be monitored for lambda values 0.05 and
    0.1.

    To see which monitoring computeMetrics are implemented use:

    >>> help(antco.report.AVAILABLE_METRICS)

    """
    def __init__(self, metrics: dict = None):
        self._metrics = {}

        if metrics is not None:
            for metric, metric_args in metrics.items():
                assert metric.lower() in AVAILABLE_METRICS, 'Metric %s not available.' % metric
                self._metrics[METRICS[metric.lower()]] = metric_args

        self._values = defaultdict(dict)
        self._best_solution = {'solution': None, 'score': None, 'iteration': None, 'ant': None}

    def __repr__(self):
        return f'Report'

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self._values)

    @property
    def values(self) -> dict:
        """
        Returns a dictionary with all the values stored during the execution of the algorithm.

        Returns
        -------
        :dict
            :key int
                Iteration number.
        """
        return self._values

    @property
    def best_solution(self) -> dict:
        """
        Return a dictionary with the best solution found by the algorithm.

        Returns
        -------
        :return dict
            :key "ant": Best Ant instance.
            :key "solution": Nodes visited by the ant.
            :key "score": Score associated to the ant.
            :key "iteration": Iteration in which the best solution was found.
        """
        return self._best_solution

    def computeMetrics(self, iteration: int, pheromones: np.ndarray, adj_matrix: np.ndarray,
                       name: str = None):
        """ 
        Function that computes the computeMetrics specified in the constructor.

        Parameters
        ----------
        iteration: int
            Current iteration.

        name: str
            Name preceding the names returned by the particular metric. For more information about
            the computeMetrics use help(antco.report.METRICS[METRIC_NAME]). To see the name of the computeMetrics
            that are available use help(antco.report.AVAILABLE_METRICS).

        pheromones: np.ndarray of shape (n_nodes, n_nodes), dtype=np.float64
            Pheromone matrix to be analysed.

        adj_matrix: np.ndarray of shape (n_nodes, n_nodes), dtype=np.int8
            Adjacency matrix defining the graph structure.
        """
        if len(self._metrics) > 0:
            for metric, metric_args in self._metrics.items():
                values = metric(pheromones, adj_matrix, **metric_args)

                self.save(iteration,  **{'%s_%s' % (name, key) if name is not None else key: value
                                         for key, value in values.items()})

    def get(self, keyword: str) -> dict:
        """
        Returns a dictionary where the key corresponds to the interaction number of the algorithm
        and the value to the value selected in set_cost for the argument received as parameter.

        Parameters
        ----------
        keyword: str
            Parameter from which to obtain the values saved during the execution of the algorithm.

        Returns
        -------
        :dict
            Dictionary where the key corresponds to the interaction number and the values to the
            data saved for that parameter.
         """
        if len(self._values.values()) == 0:
            assert False, 'The report object has no recorded value. For more information on how it ' \
                          'works use help(antco.report.Report).'

        assert keyword in list(self._values.values())[1], \
            'Parameter %s not found in Report. Available parameters: %r' % (keyword, list(self._values.values())[0].keys())

        return {iteration: values[keyword] for iteration, values in self._values.items()}

    def save(self, iteration: int, **kwargs):
        """
        Stores the values received as an argument associated with the interaction passed as an
        argument.

        Parameters
        ----------
        iteration: int
            Algorithm interaction number.

        kwargs:
            Parameters to be saved.
        """
        for key, value in kwargs.items():
            self._values[iteration][key] = value

    def updateBest(self, iteration: int, score: float, ant: Ant):
        """
        Method that saves the best solution found.

        Parameters
        ----------
        iteration: int
            Iteration.

        score: float
            Ant score.

        ant: Ant
            Ant instance.
        """
        self._best_solution['iteration'] = iteration
        self._best_solution['score'] = score
        self._best_solution['solution'] = ant.visited_nodes
        self._best_solution['ant'] = ant


def computeBranchingFactor(pheromones: np.ndarray, adj_matrix: np.ndarray, lambda_values: list = None,
                           **kw) -> dict:
    """ 
    Function that computes the lambda-branching factor for the specified lambda values. 

    Parameters
    ----------
    pheromones: np.ndarray of shape (n_nodes, n_nodes) and dtype np.float64
        Pheromone matrix to be analysed.

    adj_matrix: np.ndarray of shape (n_nodes, n_nodes) and dtype np.int8
        Adjacency matrix defining the graph structure.

    lambda_values: list
        List of lambda values to be computed.

    Returns
    -------
    :dict
        Dictionary where the keys will have the format 'lambda_VALUE_DE_LAMBDA' (i.e. lambda_0.05) and t
        he values will be the value obtained for the given lambda.

    """
    # Default lambda parameters
    lambda_values = lambda_values if lambda_values is not None else [0.05, 0.1, 0.2, 0.4]
    return {'lambda_%.4f' % val: getBranchingFactor(pheromones, adj_matrix, val) for val in lambda_values}


AVAILABLE_METRICS = ['branchingfactor']
METRICS = {
    AVAILABLE_METRICS[0]: computeBranchingFactor
}
