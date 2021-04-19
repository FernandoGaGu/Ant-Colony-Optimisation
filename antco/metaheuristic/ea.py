import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
from deap import base
from deap import algorithms
from deap import creator
from deap import tools
from .base import MetaHeuristic
from ..optim import ObjectiveFunction


class PermutationGA(MetaHeuristic):
    """
    Class implementing a permutation-based evolutionary strategy. This class allows to optimise the
    permutations given by the ants, for example for the traveller salesman problem: each ant selects
    a route, the best routes are received by this algorithm which is in charge of refining them.
    The new routes refined by the evolutionary strategy are used to update the values of the
    pheromone matrix.

    The evolutionary algorithm executed will correspond to the one implemented in
    deap.algorithms.eaSimple. The user must specify the cost function used to perform the mapping
    from genotype (in the form of permutation) to phenotype and evaluate its quality.

    Parameters
    ----------
    antco_objective: antco.optim.ObjectiveFunction
        Objective function defined using the antco.optim.ObjectiveFunction interface.

    genetic_objective: callable
        Objective function that will receive each individual (encoded as a list of integers without
        repetition) and will return a tuple where the first element will be the scalar value
        associated with that genotype.

    best_ants: int
        Number of the best ants to be passed to the metaheuristic strategy.

    population_size: int
        Genetic algorithm population size.

    crossover_prob: float
        Genetic algorithm cross-over probability.

    mutation_prob: float
        Genetic algorithm mutation probability.

    individual_mutation_prob: float
        Genetic algorithm genotype position mutation probability.

    generations: int
        Genetic algorithm number of generations.

    tournsize: int
        Genetic algorithm number of individuals selected for tournament selection.

    hof: int, default=None
        Genetic algorithm elite (Hall of Fame) used in deap.algorithms.eaSimple

    n_jobs: int, default=1
        Number of proccessed executed in paralel.

    genetic_objective_args: dict, default=None
        If the genetic objective function requires additional parameters these must be passed to
        the constructor in the form of a dictionary where the name of the parameter received in the
        objective function must correspond to the key and the value passed to the value.

    display_convergence: bool, default=False
        Parameter indicating whether to display the convergence graphs of the evolutionary
        algorithm at each iteration, useful for debugging and hyperparameter tuning purposes.
        Defaults to off.
    """
    def __init__(self, antco_objective: ObjectiveFunction, genetic_objective: callable,
                 best_ants: int, population_size: int, crossover_prob: float, mutation_prob: float,
                 individual_mutation_prob: float, generations: int, tournsize: int = 2,
                 hof: int = None, n_jobs: int = 1, genetic_objective_args: dict = None,
                 display_convergence: bool = False):

        super(PermutationGA, self).__init__(antco_objective, n_jobs)

        self._best_ants = best_ants
        self._generations = generations
        self._mutation_prob = mutation_prob
        self._indpb = individual_mutation_prob
        self._crossover_prob = crossover_prob
        self._population_size = population_size
        self._display_convergence = display_convergence
        self._toolbox = base.Toolbox()

        # Create fitness function
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))

        # Register GA elements
        if genetic_objective_args is None:
            self._toolbox.register('evaluate', genetic_objective)
        else:   # Pass fixed arguments to the cost function
            self._toolbox.register('evaluate', genetic_objective, **genetic_objective_args)
        self._toolbox.register('select', tools.selTournament, tournsize=tournsize)
        self._toolbox.register('mate', tools.cxOrdered)
        self._toolbox.register('mutate', tools.mutShuffleIndexes, indpb=individual_mutation_prob)

        # Create Hall of Fame
        self._hof = tools.HallOfFame(hof) if hof is not None else hof

        # Create stats
        if self._display_convergence:
            self._stats = tools.Statistics(lambda ind: ind.fitness.values)
            self._stats.register('max', np.max)
            self._stats.register('mean', np.mean)
        else:
            self._stats = None

        # Parallel execution
        if n_jobs > 1 or n_jobs == -1:
            pool = mp.Pool(mp.cpu_count() if n_jobs == -1 else n_jobs)
            self._toolbox.register("map", pool.map)

    def __repr__(self):
        return f'PermutationGA(best_ants={self._best_ants}, generations={self._generations}, ' \
               f'population_size={self._population_size}, mutation_prob={self._mutation_prob}, ' \
               f'crossover_prob={self._crossover_prob}, individual_mutation_prob={self._indpb})'

    def __str__(self):
        return self.__repr__()

    def optimise(self, ants: list, scores: list) -> list:
        # Get the index associated with the best ants
        best_ants = np.argsort(scores)[::-1][:self._best_ants]

        # Initialisation of the genotypes of the individuals using the best routes found by the ACO
        initial_genotypes = [ants[idx].visited_nodes.tolist() for idx in best_ants]
        self.create_permutation_ind(self._toolbox, creator.FitnessMax, initial_genotypes)

        # Create initial population
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        population = self._toolbox.population(n=self._population_size)

        # Execute algorithm
        if self._display_convergence:
            population, logbook = algorithms.eaSimple(
                population, self._toolbox, cxpb=self._crossover_prob, mutpb=self._mutation_prob,
                ngen=self._generations, stats=self._stats, halloffame=self._hof, verbose=False)

            plt.plot(logbook.select('max'), color='red')
            plt.plot(logbook.select('mean'), color='green')
            plt.xlabel('Generation')
            plt.ylabel('Max / Average Fitness')
            plt.title('Max and Average fitness over Generations')
            plt.show()
        else:
            population, logbook = algorithms.eaSimple(
                population, self._toolbox, cxpb=self._crossover_prob, mutpb=self._mutation_prob,
                ngen=self._generations, stats=self._stats, halloffame=self._hof, verbose=False)
        # Unregister elements
        del creator.Individual
        self._toolbox.unregister('individual')
        self._toolbox.unregister('attr_permutation')

        # Create Ants (when Hall of Fame hasn't been specified)
        if self._hof is None:
            improved_ants = [ants[0].new() for _ in range(self._best_ants)]
            fitness_values = [ind.fitness.values[0] for ind in population]
            best_fitness = np.argsort(fitness_values)[::-1][:self._best_ants]
            for idx, ind_idx in enumerate(best_fitness):
                improved_ants[idx].getVisitedNodes = population[ind_idx]
        # Create Ants (using Hall of Fame )
        else:
            improved_ants = [ants[0].new() for _ in range(len(self._hof.items))]
            for idx, solution in enumerate(self._hof.items):
                improved_ants[idx].visited_nodes = solution

        return improved_ants

    @classmethod
    def create_permutation_ind(cls, toolbox: base.Toolbox, fitness_function: callable,
                               initial_values: list = None, individual_size: int = None):
        """
        Method that allows to create and register (following the guidelines defined in DEAP) the
        genotype of the individuals (registered as 'Individual') and the generating function of
        individuals (registered as 'individual').

        Parameters
        ----------
        toolbox: base.Toolbox
            DEAP Toolbox instance.

        fitness_function: callable
            DEAP fitness function.

        initial_values: list, default=None
            List of list of initial genotypes used for the creation of the initial population, this
            allows incorporating a priori knowledge about better solutions and usually gives better
            results than random initialisation of the genotypes.

            If this parameter is not provided, it will be necessary to provide tha argument
            individual_size.

        individual_size: int, default=None
            Size of the individual genotype.

            If this parameter is not provided, it will be necessary to provide tha argument
            individual_size.

        Notes
        -----
        Parameters initial_values and individual_size cannot be provided at the same time.
        """
        assert (initial_values is None or individual_size is None) and \
               not (initial_values is None and individual_size is None), \
               'Either the initial_values or individual_size must be provided.'

        # Create from initial values
        if initial_values is not None:
            ind_generator = lambda initial_values: initial_values[
                random.randint(0, len(initial_values) - 1)]
            toolbox.register('attr_permutation', ind_generator, initial_values)
        # Create randomly
        else:
            toolbox.register('attr_permutation', random.sample, range(individual_size), individual_size)

        creator.create('Individual', list, fitness=fitness_function)
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_permutation)


class SetGA(MetaHeuristic):
    """
    Class implementing a integer-based evolutionary strategy. This class allows optimisation to be
    performed on a genotype of non-repeating integer values of variable length.

        Parameters
    ----------
    antco_objective: antco.optim.ObjectiveFunction
        Objective function defined using the antco.optim.ObjectiveFunction interface.

    genetic_objective: callable
        Objective function that will receive each individual (encoded as a list of integers without
        repetition) and will return a tuple where the first element will be the scalar value
        associated with that genotype.

    best_ants: int
        Number of the best ants to be passed to the metaheuristic strategy.

    population_size: int
        Genetic algorithm population size.

    crossover_prob: float
        Genetic algorithm cross-over probability.

    mutation_prob: float
        Genetic algorithm mutation probability.

    generations: int
        Genetic algorithm number of generations.

    tournsize: int
        Genetic algorithm number of individuals selected for tournament selection.

    hof: int, default=None
        Genetic algorithm elite (Hall of Fame) used in deap.algorithms.eaSimple

    n_jobs: int, default=1
        Number of proccessed executed in paralel.

    genetic_objective_args: dict, default=None
        If the genetic objective function requires additional parameters these must be passed to
        the constructor in the form of a dictionary where the name of the parameter received in the
        objective function must correspond to the key and the value passed to the value.

    display_convergence: bool, default=False
        Parameter indicating whether to display the convergence graphs of the evolutionary
        algorithm at each iteration, useful for debugging and hyperparameter tuning purposes.
        Defaults to off.
    """
    def __init__(self, antco_objective: ObjectiveFunction, genetic_objective: callable,
                 best_ants: int, population_size: int, crossover_prob: float, mutation_prob: float,
                 generations: int, tournsize: int = 2, hof: int = None, n_jobs: int = 1,
                 genetic_objective_args: dict = None, display_convergence: bool = False):

        super(SetGA, self).__init__(antco_objective, n_jobs)

        self._best_ants = best_ants
        self._generations = generations
        self._mutation_prob = mutation_prob
        self._crossover_prob = crossover_prob
        self._population_size = population_size
        self._display_convergence = display_convergence
        self._toolbox = base.Toolbox()

        # Create fitness function
        creator.create('FitnessMax', base.Fitness, weights=(1.0,))

        # Register GA elements
        if genetic_objective_args is None:
            self._toolbox.register('evaluate', genetic_objective)
        else:   # Pass fixed arguments to the cost function
            self._toolbox.register('evaluate', genetic_objective, **genetic_objective_args)
        self._toolbox.register('select', tools.selTournament, tournsize=tournsize)
        self._toolbox.register('mate', cxSet)

        # Create Hall of Fame
        self._hof = tools.HallOfFame(hof) if hof is not None else hof

        # Create stats
        if self._display_convergence:
            self._stats = tools.Statistics(lambda ind: ind.fitness.values)
            self._stats.register('max', np.max)
            self._stats.register('mean', np.mean)
        else:
            self._stats = None

        # Parallel execution
        if n_jobs > 1 or n_jobs == -1:
            pool = mp.Pool(mp.cpu_count() if n_jobs == -1 else n_jobs)
            self._toolbox.register("map", pool.map)

    def __repr__(self):
        return f'SetGA(best_ants={self._best_ants}, generations={self._generations}, ' \
               f'population_size={self._population_size}, mutation_prob={self._mutation_prob}, ' \
               f'crossover_prob={self._crossover_prob})'

    def __str__(self):
        return self.__repr__()

    def optimise(self, ants: list, scores: list) -> list:
        # Get the index associated with the best ants
        best_ants = np.argsort(scores)[::-1][:self._best_ants]
        possible_values = np.unique([   # Select unique nodes
            node for idx in best_ants for node in ants[idx].visited_nodes]).tolist()

        # Register mutation operator
        self._toolbox.register('mutate', mutSet, possible_values=possible_values)

        # Initialisation of the genotypes of the individuals using the list of possible values
        self.create_set_ind(self._toolbox, creator.FitnessMax, possible_values)

        # Create initial population
        self._toolbox.register('population', tools.initRepeat, list, self._toolbox.individual)
        population = self._toolbox.population(n=self._population_size)

        # Execute algorithm
        if self._display_convergence:
            population, logbook = algorithms.eaSimple(
                population, self._toolbox, cxpb=self._crossover_prob, mutpb=self._mutation_prob,
                ngen=self._generations, stats=self._stats, halloffame=self._hof, verbose=False)

            plt.plot(logbook.select('max'), color='red')
            plt.plot(logbook.select('mean'), color='green')
            plt.xlabel('Generation')
            plt.ylabel('Max / Average Fitness')
            plt.title('Max and Average fitness over Generations')
            plt.show()
        else:
            population, logbook = algorithms.eaSimple(
                population, self._toolbox, cxpb=self._crossover_prob, mutpb=self._mutation_prob,
                ngen=self._generations, stats=self._stats, halloffame=self._hof, verbose=False)

        # Unregister elements
        del creator.Individual
        self._toolbox.unregister('individual')
        self._toolbox.unregister('attr_set')
        self._toolbox.unregister('mutate')

        # Create Ants (when Hall of Fame hasn't been specified)
        if self._hof is None:
            improved_ants = [ants[0].new() for _ in range(self._best_ants)]
            fitness_values = [ind.fitness.values[0] for ind in population]
            best_fitness = np.argsort(fitness_values)[::-1][:self._best_ants]
            for idx, ind_idx in enumerate(best_fitness):
                improved_ants[idx].getVisitedNodes = list(population[ind_idx])
        # Create Ants (using Hall of Fame )
        else:
            improved_ants = [ants[0].new() for _ in range(len(self._hof.items))]
            for idx, solution in enumerate(self._hof.items):
                improved_ants[idx].getVisitedNodes = list(solution)

        return improved_ants

    @classmethod
    def create_set_ind(cls, toolbox: base.Toolbox, fitness_function: callable,
                       possible_values: list):
        """
        Method to create the genotype of the individuals represented by a Set from the list of
        possible values received as an argument.

        Parameters
        ----------
        toolbox: base.Toolbox
            DEAP Toolbox instance.

        fitness_function: callable
            DEAP fitness function.

        possible_values: list
            List of possible values to insert into the individual.
        """
        creator.create('Individual', set, fitness=fitness_function)
        toolbox.register('attr_set', initSetGenotype, possible_values=list(possible_values))
        toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.attr_set)


def cxSet(ind1, ind2):
    """
    Apply a crossover operation on input sets. The first child is the intersection of the two sets,
    the second child is the difference of the two sets.

    Parameters
    ----------
    ind1: Set
        Parent 1.

    ind2: Set
        Parent 1

    Returns
    -------
    :tuple
        :[0]: Set
            Child 1.
        :[1]: Set
            Child 2.
    """
    temp = set(ind1)                # Used in order to keep type
    ind1 &= ind2                    # Intersection (inplace)
    ind2 ^= temp                    # Symmetric Difference (inplace)
    return ind1, ind2


def mutSet(individual, possible_values: list):
    """
    Mutation that pops or add an element.

    Parameters
    ----------
    individual: Set
        Individual to be mutated.

    possible_values: list
        List of possible values to insert into the individual.

    Returns
    -------
    :tuple
        :[0]: Set
            Mutated individual.
    """
    if random.random() < 0.5:
        if len(individual) > 0:     # We cannot pop from an empty set
            individual.remove(random.choice(sorted(tuple(individual))))
    else:
        individual.add(random.choice(list(possible_values)))
    return individual,


def initSetGenotype(possible_values: list):
    """
    Function to initialise the genotype of individuals represented as a Set from a list of possible
    values.

    Parameters
    ----------
    possible_values: list
        List of possible values to insert into the individual.

    Returns
    -------
    :list
        Individual genotype.

    Notes
    -----
    There must be at least two different values in possible_values.
    """
    num_elements = random.randint(2, len(possible_values) - 1)
    return np.random.choice(possible_values, size=num_elements, replace=False).tolist()
