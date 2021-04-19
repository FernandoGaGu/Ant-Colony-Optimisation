# Module containing the main Ant-Colony Optimisation algorithms implemented.
#
# Author: Fernando García Gutiérrez
# Email: fegarc05@ucm.es
#
import numpy as np
import sys
import random
import warnings
from copy import deepcopy
from .metaheuristic.base import MetaHeuristic
from .report import Report
from .ant import fixedPositions, deleteInitialPosition
from .base import DecaySchedule
from .aco import ACO, evaluateAnts, generatePaths

warnings.filterwarnings('error')


def updateReportWithBest(ants: list, scores: np.ndarray, best_score: float, report: Report,
                         iteration: int) -> float:
    """
    Function that updates the best solution of the report instance.

    Parameters
    ----------
    ants: list
        List of Ant instances.

    scores: np.ndarray, (len(ants)), dtype=np.float64
        Scores associated with the ants.

    best_score: float
        Best score seen until the current iteration.

    report: Report
        Report instance to be updated.

    iteration: int
        Current iteration.

    Returns
    -------
    :float
        Updated best score value.

    Notes
    -----
    This function will modify the internal state of the Report instance adding the best ant.
    """
    max_score = np.max(scores)

    # Update best score
    if max_score > best_score:
        best_score = max_score
        idx = np.argmax(scores)
        report.updateBest(iteration, best_score, deepcopy(ants[idx]))

    return best_score


def basic(aco_obj: ACO, parallel_evaluation: bool = False, scores_decay: DecaySchedule = None,
          evaporation_decay: DecaySchedule = None, report: Report = None,
          save_pheromones: bool = True, verbose: bool = True) -> Report:
    """
    Function that executes a simple Ant Colony Optimization algorithm based on the parameters
    defined in the ACO object.

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
    updatePheromones = aco_obj.updatePheromones
    objectiveFunction = aco_obj.objectiveFunction
    tol = aco_obj.tol
    fixed = aco_obj.fixed_positions
    accessory_node = aco_obj.accessory_node
    Q = aco_obj.Q
    R = aco_obj.R
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

        ants = generatePaths(   # Graph exploration
            ants=ants, graph=graph, H=H, P=P, alpha=alpha, beta=beta, Q=Q, R=R, n_jobs=n_jobs,
            exp_heuristic=False)

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


def bagOfAnts(aco_obj: ACO, bag_size: int, out_of_bag_size: int = 0, parallel_evaluation: bool = False,
              scores_decay: DecaySchedule = None, evaporation_decay: DecaySchedule = None,
              report: Report = None, save_pheromones: bool = True, verbose: bool = True) -> Report:
    """
    Function that executes an Ant Colony Optimization algorithm based on the parameters defined in
    the ACO object. This version of the algorithm maintains an bag_size aka a bag of ants (of the size
    specified in the execution parameters) that is preserved with the best ants. This bag will be
    the one used to update the values of the pheromone matrix (therefore, note that if an elitist
    strategy is used to update the values of the pheromone matrix, the size of the bag_size passed
    as an argument to the function and of the bag_size used to update the values of the pheromone
    matrix must be congruent).

    Parameters
    ----------
    aco_obj: antco.ACO
        Initialised antco.ACO instance, for more information on this type of object use:
        help(antco.ACO)

    bag_size: int
        Elite size maintained and used to update pheromone matrix values.

    out_of_bag_size: int, default=0
        Number of non-elite ants from the current iteration to include in the bag of ants.

    parallel_evaluation: bool, default=False  [IN TESTING]
        Parameter indicating whether to perform a parallel evaluation of the cost function. When
        the compuational cost of the function is high the performance will increase, however, if the
        compuational cost is moderate this will result in a lower performance of the algorithm.

    scores_decay: antco.optim.DecaySchedule subclass, default=None
        Instance of a antco.optim.DecaySchedule representing a decay schedule that will applied to
        the score values with which the pher200omone values are updated in each iteration.
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

        >>> report = bagOfAnts(...)
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
    updatePheromones = aco_obj.updatePheromones
    objectiveFunction = aco_obj.objectiveFunction
    tol = aco_obj.tol
    fixed = aco_obj.fixed_positions
    accessory_node = aco_obj.accessory_node
    Q = aco_obj.Q
    R = aco_obj.R
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
    ant_scores, norm_scores = np.empty(shape=len(ants)), np.empty(shape=bag_size+out_of_bag_size)
    bag_of_ants = None
    boa_scores = np.full(shape=bag_size+out_of_bag_size, fill_value=-999_999.999)

    current_iteration = 1
    it_without_improvements = 0
    best_score = -float('inf')
    while current_iteration <= iterations:

        ants = generatePaths(   # Graph exploration
            ants=aco_obj.ants, graph=graph, H=H, P=P, alpha=alpha, beta=beta, Q=Q, R=R, n_jobs=n_jobs,
            exp_heuristic=False)

        # Evaluate ant paths using the objectiveFunction function (it will be maximized)
        ant_scores = evaluateAnts(
            ants=ants, objectiveFunction=objectiveFunction, parallel_evaluation=parallel_evaluation,
            n_jobs=n_jobs)

        for ant in ants:
            if ant.initial_position is None:
                print('None after generatePaths')

        # HACK: Elite upgrade (the current implementation is inefficient)
        if bag_of_ants is None:
            ordered_scores = np.argsort(ant_scores)[::-1]   # Ascending order
            bag_of_ants = [ants[ordered_scores[i]] for i in range(bag_size + out_of_bag_size)]
            boa_scores = ant_scores[ordered_scores][:bag_size + out_of_bag_size]
            boa_scores = np.array(boa_scores)
        else:
            all_scores = np.append(boa_scores[:bag_size], ant_scores)
            all_ants = bag_of_ants[:bag_size] + ants
            ordered_scores = np.argsort(all_scores)[::-1]   # Ascending order
            bag_of_ants = [all_ants[ordered_scores[i]] for i in range(bag_size + out_of_bag_size)]
            boa_scores = all_scores[ordered_scores][:bag_size + out_of_bag_size]

        # Update best score and save best solution
        new_best = updateReportWithBest(
            ants=bag_of_ants, scores=boa_scores, best_score=best_score, report=report,
            iteration=current_iteration)

        if new_best > best_score:
            best_score = new_best
            it_without_improvements = 0
        else:
            it_without_improvements += 1

        # Scale scores
        if scaleScores is not None:
            norm_scores[:] = boa_scores[:]  # Copy scores
            norm_scores = scaleScores(norm_scores, best_score)
        else:
            norm_scores = boa_scores

        # Update pheromones according to the scores
        updatePheromones(
            paths=np.array([ant.adj_matrix for ant in bag_of_ants], dtype=np.int8), P=P,
            ant_scores=norm_scores if scores_decay is None else norm_scores * scores_decay(current_iteration),
            rho=rho if evaporation_decay is None else rho * evaporation_decay(current_iteration),
            **pheromone_update_kw)

        # Compute convergence statistics
        mean_scores = np.mean(boa_scores)
        min_score = boa_scores[bag_size + out_of_bag_size - 1]
        max_score = boa_scores[0]
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


def hybrid(aco_obj: ACO, metaheuristic: MetaHeuristic, apply_meta_each: int = 1,
           parallel_evaluation: bool = False, scores_decay: DecaySchedule = None,
           evaporation_decay: DecaySchedule = None, report: Report = None,
           save_pheromones: bool = True, verbose: bool = True) -> Report:
    """
    Function that executes an Ant Colony Optimization hybrid algorithm (ACO + another metaheuristic
    strategy) based on the parameters defined in the ACO object.

    Parameters
    ----------

    aco_obj: antco.ACO
        Initialised antco.ACO instance, for more information on this type of object use:
        help(antco.ACO)

    metaheuristic: antco.metaheuristic.base.MetaHeuristic subclass
        Instance following the interface defined in antco.extensions.MetaHeuristic. For more info
        use help(antco.extensions.MetaHeuristic).

    apply_meta_each: int, default=1
        Parameter indicating how many generations the metaheuristic will be applied for the
        refinement of the solutions given by the ants.

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

        >>> report  = hybrid(...)
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
    updatePheromones = aco_obj.updatePheromones
    objectiveFunction = aco_obj.objectiveFunction
    tol = aco_obj.tol
    fixed = aco_obj.fixed_positions
    accessory_node = aco_obj.accessory_node
    Q = aco_obj.Q
    R = aco_obj.R
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

    current_iteration = 1
    it_without_improvements = 0
    best_score = -float('inf')
    while current_iteration <= iterations:

        ants = generatePaths(   # Graph exploration
            ants=ants, graph=graph, H=H, P=P, alpha=alpha, beta=beta, Q=Q, R=R, n_jobs=n_jobs,
            exp_heuristic=False)

        if current_iteration % apply_meta_each == 0:  # Apply metaheuristic
            ants, ant_scores = metaheuristic(ants)
        else:  # Conventional evaluation
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
            norm_scores = deepcopy(ant_scores)  # Copy scores
            norm_scores = scaleScores(norm_scores, best_score)
        else:
            norm_scores = ant_scores

        # Update pheromones according to the scores
        updatePheromones(
            paths=np.array([ant.adj_matrix for ant in ants], dtype=np.int8), P=P,
            ant_scores=norm_scores if scores_decay is None else norm_scores * scores_decay(current_iteration),
            rho=rho if evaporation_decay is None else rho * evaporation_decay(current_iteration),
            **pheromone_update_kw)

        # Compute convergence statistics
        mean_scores = np.mean(ant_scores)
        min_score = np.min(ant_scores)
        max_score = np.max(ant_scores)
        report.save(current_iteration, mean_cost=mean_scores, max_cost=max_score)

        if save_pheromones:  # Save pheromone values
            report.save(current_iteration, pheromones=deepcopy(P))

        # Compute monitoring computeMetrics
        report.computeMetrics(current_iteration, P, graph)

        # Reposition the ants
        ants = aco_obj.ants
        if fixed and accessory_node is None:  # Put ants in fixed positions (if specified)
            fixedPositions(ants, graph)
        # If an accessory node has been specified, place the ants on that node
        if accessory_node is not None:
            for ant in ants:
                ant.setInitialPosition(accessory_node)

        # After several generations without improvements, do an early stopping of the algorithm.
        if it_without_improvements > tol: break

        if verbose:
            sys.stdout.write('\rCost: Mean: %.4f (Min: %.4f Max: %.4f) (iteration %d)' %
                             (float(mean_scores), min_score, max_score, current_iteration))

        current_iteration += 1

    return report
