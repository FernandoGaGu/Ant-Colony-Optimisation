import sys; sys.path.append('..')
import antco

# Min-Max Ant System (MMAS)
elite_mmas = 2
pher_init_mmas = 5.0
evaporation_mmas = 0.05
limits_mmas = (0.5, 10.0)
weight_mmas = 0.6


checker = antco.tools.HyperparameterCheckerMMAS(
    n_ants=elite_mmas, pher_init=pher_init_mmas, evaporation=evaporation_mmas, limits=limits_mmas,
    weight=weight_mmas)
checker.plot()

# Ant System (AS)
elite_as = 20
pher_init_as = 5.0
evaporation_as = 0.05
weight_as = 0.15

checker = antco.tools.HyperparameterCheckerAS(
    n_ants=elite_as, pher_init=pher_init_as, evaporation=evaporation_as, weight=weight_as)
checker.plot()
