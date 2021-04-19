# Module containing the experimental features of the framework that are still under development.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import sys
import random
import warnings
from copy import deepcopy

from ..report import Report
from ..ant import fixedPositions, deleteInitialPosition
from ..base import DecaySchedule
from ..aco import ACO, evaluateAnts, generatePathsACS
from ..algorithm import updateReportWithBest

warnings.filterwarnings('error')


def antColonySystem(aco_obj: ACO, parallel_evaluation: bool = False, scores_decay: DecaySchedule = None,
                    evaporation_decay: DecaySchedule = None, report: Report = None,
                    save_pheromones: bool = True, verbose: bool = True) -> Report:
    """
    Function that executes the Ant Colony System algorithm based on

        Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning
        approach to the traveling salesman problem. IEEE Transactions on evolutionary computation,
        1(1), 53-66.


    Parameters
    ----------
    aco_obj: antco.ACO
        Initialised antco.ACO instance, for more information on this type of object use:
        help(antco.ACO)

    parallel_evaluation: bool, default=False  [IN TESTING]
        Parameter indicating whether to perform a parallel evaluation of the cost function. When
        the compuational cost of the function is high the performance will increase, however, if the
        compuational cost is moderate this will result in a lower performance of the algorithm.

    scores_decay: antco.optim.DecaySchedule subclass, default=None
        Instance of a antco.optim.DecaySchedule representing a decay schedule that will applied to
        the score values with which the pheromone values are updated in each iteration.
        For more info use help(antco.optim.DecaySchedule).

    evaporation_decay: antco.optim.DecaySchedule subclass, default=None
        Same as scores_decay, in this case used to update the evaporation rate value.
        For more info use help(antco.optim.DecaySchedule).

    report: antco.report.Report, default=None
        antco.report.Report instance, by default a report will be initialised.

    save_pheromones: bool, default=True
        Value indicating whether to store in the report the values of the pheromone matrix at each
        iteration. If True these may be retrieved by:

            >>> report.get('pheromones')

    verbose: bool, default=True
        Indicates whether to show convergence information during the execution of the algorithm.

    Returns
    -------
    :antco.report.Report
        Returns a antco.report.Report instance with information about the convergence of the
        algorithm. The convergence parameters can be accessed via:

        >>> report = antco.algorithm.basic(...)
        >>> report.get('min_cost')
        >>> report.get('mean_cost')
        >>> report.get('max_cost')

        For more info about Report instances use help(antco.report.Report).
    """
    # Get parameters from aco_obj
    seed = aco_obj.seed
    graph = aco_obj.graph
    iterations = aco_obj.iterations
    n_jobs = aco_obj.n_jobs
    ants = aco_obj.ants
    H = aco_obj.heuristic
    P = aco_obj.pheromones
    alpha = aco_obj.alpha
    beta = aco_obj.beta
    if aco_obj.precompute_heuristic:
        H = np.power(H, beta)
    pheromone_update_kw = aco_obj.pheromone_update_kw
    rho = pheromone_update_kw['rho']; del pheromone_update_kw['rho']  # Get evaporation parameter
    decay = pheromone_update_kw['decay']; del pheromone_update_kw['decay']   # Get decay parameter
    pher_init_val = aco_obj.pher_init_val
    updatePheromones = aco_obj.updatePheromones
    objectiveFunction = aco_obj.objectiveFunction
    tol = aco_obj.tol
    fixed = aco_obj.fixed_positions
    accessory_node = aco_obj.accessory_node
    Q = aco_obj.Q
    scaleScores = aco_obj.scaleScores

    if report is None:  # Initialize a Report instance (if not provided)
        report = Report({'BranchingFactor': {'lambda_values': [0.5]}})

    if seed is not None:  # Set random state
        random.seed(seed)
        np.random.seed(seed)

    if fixed and accessory_node is None:  # Put ants in fixed positions (if specified)
        fixedPositions(ants, graph)

    # If an accessory node has been specified, place the ants on that node
    if accessory_node is not None:
        for ant in ants:
            ant.setInitialPosition(accessory_node)

    # Pre-compute the array to store the ant scores
    ant_scores, norm_scores = np.empty(shape=len(ants)), np.empty(shape=len(ants))

    current_iteration = 1
    it_without_improvements = 0
    best_score = -float('inf')
    while current_iteration <= iterations:

        ants = generatePathsACS(   # Graph exploration
            ants=ants, graph=graph, H=H, P=P, alpha=alpha, beta=beta, decay=decay,
            pher_init=pher_init_val, Q=Q, exp_heuristic=False)

        # Evaluate ant paths using the objectiveFunction function (it will be maximized)
        ant_scores = evaluateAnts(
            ants=ants, objectiveFunction=objectiveFunction, parallel_evaluation=parallel_evaluation,
            n_jobs=n_jobs)

        # Update best score and save best solution
        new_best = updateReportWithBest(
            ants=ants, scores=ant_scores, best_score=best_score, report=report,
            iteration=current_iteration)

        if new_best > best_score:
            best_score = new_best
            it_without_improvements = 0
        else:
            it_without_improvements += 1

        # Scale scores
        if scaleScores is not None:
            norm_scores[:] = ant_scores[:]  # Copy scores
            norm_scores = scaleScores(norm_scores, best_score)
        else:
            norm_scores = ant_scores

        # Update pheromones according to the scores
        updatePheromones(
            paths=np.array([ant.adj_matrix for ant in ants], dtype=np.int8), P=P,
            ant_scores=norm_scores if scores_decay is None else norm_scores * scores_decay(current_iteration),
            rho=rho if evaporation_decay is None else rho * evaporation_decay(current_iteration),
            **pheromone_update_kw)

        if not fixed and accessory_node is None:  # Restart ants initial position
            deleteInitialPosition(ants)

        # Compute convergence statistics
        mean_scores = np.mean(ant_scores)
        min_score = np.min(ant_scores)
        max_score = np.max(ant_scores)
        report.save(current_iteration, mean_cost=mean_scores, max_cost=max_score)

        if save_pheromones:  # Save pheromone values
            report.save(current_iteration, pheromones=deepcopy(P))

        # Compute monitoring computeMetrics
        report.computeMetrics(current_iteration, P, graph)

        # After several generations without improvements, do an early stopping of the algorithm.
        if it_without_improvements > tol: break

        if verbose:
            sys.stdout.write('\rCost: Mean: %.4f (Min: %.4f Max: %.4f) (iteration %d)' %
                             (float(mean_scores), min_score, max_score, current_iteration))

        current_iteration += 1

    return report


