import numpy as np
from copy import deepcopy
from antco import (
    updateUndAS,
    updateDirAS,
    updateUndMMAS,
    updateDirMMAS,
    updateUndEliteMMAS,
    updateDirEliteMMAS,
    updateDirEliteAS,
    updateUndEliteAS,
    updateDirLocalPher,
    updateUndLocalPher,
    updateUndACS,
    updateDirACS)

from antco import Ant


def test_directed_AS_update():
    """ antco.pheromone.updateDirAS() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)
    expected = np.array([
        [0.0, 0.9267931249792329, 0.4776117072586296, 1.6791352931971335],
        [0.9267931249792329, 0.0, 0.5591658434565883, 0.7150135839042728],
        [0.4776117072586296, 0.5591658434565883, 0.0, 1.0865920636193305],
        [1.6791352931971335, 0.7150135839042728, 1.0865920636193305, 0.0]], dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)

    updateDirAS(paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateDirAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirAS()')


def test_undirected_AS_update():
    """ antco.pheromone.updateUndAS() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T)/2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)
    expected = np.array([
        [0.0, 0.9267931249792329, 0.4776117072586296, 1.6791352931971335],
        [0.9267931249792329, 0.0, 0.5591658434565883, 0.7150135839042728],
        [0.4776117072586296, 0.5591658434565883, 0.0, 1.0865920636193305],
        [1.6791352931971335, 0.7150135839042728, 1.0865920636193305, 0.0]], dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4]).astype(np.float64)

    updateUndAS(paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateUndAS()')


def test_directed_AS_elite_update():
    """ antco.pheromone.updateDirEliteAS() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 1, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.6414344987114436, 0.6820893643835099, 1.2433082310436099],
        [0.6414344987114436, 0.0, 0.4473326730988265, 0.5720108649925117],
        [0.3820893643835099, 0.4473326730988265, 0.0, 0.7692736491472838],
        [1.2433082310436099, 0.5720108649925117, 0.7692736491472838, 0.0]],
        dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)

    updateDirEliteAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, elite=2, weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.directed_AS_elite__update()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirEliteAS()')


def test_undirected_AS_elite_update():
    """ antco.pheromone.updateUndEliteAS() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T)/2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 1, 1],
         [1, 0, 0, 0],
         [1, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)
    expected = np.array([
        [0.0, 0.6414344987114436, 0.3820893643835099, 1.2433082310436099], 
        [0.6414344987114436, 0.0, 0.7473326730988266, 0.5720108649925117], 
        [0.3820893643835099, 0.7473326730988266, 0.0, 0.7692736491472838], 
        [1.2433082310436099, 0.5720108649925117, 0.7692736491472838, 0.0]],
        dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4]).astype(np.float64)

    updateUndEliteAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, elite=2, weight=1.0)
    
    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndEliteAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateUndEliteAS()')


def test_directed_MMAS_update():
    """ aco.pheromone.directed_mmas_update() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)
    expected = np.array([
        [0.0, 0.4267931249792329, 0.4776117072586296, 1.1791352931971335],
        [0.4267931249792329, 0.0, 0.5591658434565883, 0.3150135839042728],
        [0.4776117072586296, 0.5591658434565883, 0.0, 0.5865920636193305],
        [1.1791352931971335, 0.7150135839042728, 0.5865920636193305, 0.0]], np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)

    updateDirMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateDirMMAS()'

    expected2 = np.array([
        [0.0, 0.34143449871144366, 0.3820893643835099, 1.1433082310436098],
        [0.34143449871144366, 0.0, 0.4473326730988265, 0.2520108661846046],
        [0.3820893643835099, 0.4473326730988265, 0.0, 0.6692736491472838],
        [1.1433082310436098, 0.7720108649925117, 0.6692736491472838, 0.0]], np.float64)

    updateDirMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), weight=0.5)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected2, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateDirMMAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirMMAS()')


def test_directed_MMAS_elite_update():
    """ aco.pheromone.directed_mmas_elite_update() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 1, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.6414344987114436, 0.6820893643835099, 1.2433082310436099],
        [0.6414344987114436, 0.0, 0.4473326730988265, 0.2520108661846046],
        [0.3820893643835099, 0.4473326730988265, 0.0, 0.7692736491472838],
        [1.2433082310436099, 0.5720108649925117, 0.7692736491472838, 0.0]], np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)

    updateDirEliteMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), elite=2,
        weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateDirEliteMMAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirEliteMMAS()')


def test_undirected_MMAS_update():
    """ aco.pheromone.undirected_mmas_update() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.4267931249792329, 0.4776117072586296, 1.1791352931971335], 
        [0.4267931249792329, 0.0, 0.5591658434565883, 0.7150135839042728], 
        [0.4776117072586296, 0.5591658434565883, 0.0, 0.5865920636193305], 
        [1.1791352931971335, 0.7150135839042728, 0.5865920636193305, 0.0]], 
        dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)

    updateUndMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndMMAS()'

    expected2 = np.array([
        [0.0, 0.34143449871144366, 0.3820893643835099, 1.1433082310436098],
        [0.34143449871144366, 0.0, 0.4473326730988265, 0.7720108649925117],
        [0.3820893643835099, 0.4473326730988265, 0.0, 0.6692736491472838],
        [1.1433082310436098, 0.7720108649925117, 0.6692736491472838, 0.0]],
        dtype=np.float64)

    updateUndMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), weight=0.5)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected2, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndMMAS()'

    print('SUCCESSFUL TEST: antco.pheromone.undirected_mmas_update()')


def test_undirected_MMAS_elite_update():
    """ antco.pheromone.updateUndEliteMMAS() unit testing """
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.6414344987114436, 0.3820893643835099, 1.2433082310436099], 
        [0.6414344987114436, 0.0, 0.7473326730988266, 0.5720108649925117], 
        [0.3820893643835099, 0.7473326730988266, 0.0, 0.7692736491472838], 
        [1.2433082310436099, 0.5720108649925117, 0.7692736491472838, 0.0]],
        dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 0.4], dtype=np.float64)
    
    updateUndEliteMMAS(
        paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, limits=(0, 2), elite=2, weight=1.0)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndEliteMMAS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateUndEliteMMAS()')


def test_undirected_local_update():
    np.random.seed(1997)
    decay = 0.2
    init_val = 1.0
    P = np.random.uniform(low=1.0, high=3.0, size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P, 0)
    P = (P + P.T) / 2  # Symmetric matrix
    P_t0 = deepcopy(P)

    ant1 = Ant(l_min=0, l_max=5, graph_type='u'); ant1.initAdjMatrix(4)
    ant2 = Ant(l_min=0, l_max=5, graph_type='u'); ant2.initAdjMatrix(4)
    ant3 = Ant(l_min=0, l_max=5, graph_type='u'); ant3.initAdjMatrix(4)
    ant1.visited_nodes = [0, 2, 3]
    ant2.visited_nodes = [0, 1, 2]
    ant3.visited_nodes = [3, 0, 2]

    updateUndLocalPher(ant1, P, decay, init_val)
    assert P_t0[0, 0] == P[0, 0], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[2, 3] > P[2, 3], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[3, 2] > P[3, 2], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    P_t0 = deepcopy(P)
    updateUndLocalPher(ant2, P, decay, init_val)
    assert P_t0[2, 3] == P[2, 3], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[1, 2] > P[1, 2], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[2, 1] > P[2, 1], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    P_t0 = deepcopy(P)
    updateUndLocalPher(ant3, P, decay, init_val)
    assert P_t0[1, 2] == P[1, 2], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[0, 2] > P[0, 2], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'
    assert P_t0[2, 0] > P[2, 0], 'FAILED TEST: antco.pheromone.updateUndLocalPher()'

    print('SUCCESSFUL TEST: antco.pheromone.updateUndLocalPher()')


def test_directed_local_update():
    np.random.seed(1997)
    decay = 0.2
    init_val = 1.0
    P = np.random.uniform(low=1.0, high=3.0, size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P, 0)
    P = (P + P.T) / 2  # Symmetric matrix
    P_t0 = deepcopy(P)

    ant1 = Ant(l_min=0, l_max=5, graph_type='d'); ant1.initAdjMatrix(4)
    ant2 = Ant(l_min=0, l_max=5, graph_type='u'); ant2.initAdjMatrix(4)
    ant3 = Ant(l_min=0, l_max=5, graph_type='u'); ant3.initAdjMatrix(4)
    ant1.visited_nodes = [0, 2, 3]
    ant2.visited_nodes = [0, 1, 2]
    ant3.visited_nodes = [3, 0, 2]

    updateDirLocalPher(ant1, P, decay, init_val)
    assert P_t0[0, 0] == P[0, 0], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[2, 3] > P[2, 3], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[3, 2] == P[3, 2], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    P_t0 = deepcopy(P)
    updateDirLocalPher(ant2, P, decay, init_val)
    assert P_t0[2, 3] == P[2, 3], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[1, 2] > P[1, 2], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[2, 1] == P[2, 1], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    P_t0 = deepcopy(P)
    updateDirLocalPher(ant3, P, decay, init_val)
    assert P_t0[1, 2] == P[1, 2], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[0, 2] > P[0, 2], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'
    assert P_t0[2, 0] == P[2, 0], 'FAILED TEST: antco.pheromone.updateDirLocalPher()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirLocalPher()')


def test_undirected_ACS():
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [1, 0, 0, 0],
         [0, 0, 0, 1],
         [1, 0, 1, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.5334914082114515, 0.5970146362973399, 1.1591352988595747],
        [0.5334914082114515, 0.0, 0.6989573069245543, 0.695013589566714],
        [0.5970146362973399, 0.6989573069245543, 0.0, 0.5665920692817716],
        [1.1591352988595747, 0.695013589566714, 0.5665920692817716, 0.0]],
        dtype=np.float64)

    ant_scores = np.array([0.2, 0.3, 1.9], dtype=np.float64)

    updateUndACS(paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, weight=1.0)
    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateUndACS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateUndACS()')


def test_directed_ACS():
    np.random.seed(1997)
    evaporation = 0.2
    P_t0 = np.random.uniform(size=(4, 4)).astype(np.float64)
    np.fill_diagonal(P_t0, 0)
    P_t0 = (P_t0 + P_t0.T) / 2  # Symmetric matrix
    paths = np.array([
        # Ant 1
        [[0, 1, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 0, 0]],
        # Ant 2
        [[0, 1, 0, 1],
         [1, 0, 1, 0],
         [0, 1, 0, 1],
         [1, 0, 1, 0]],
        # Ant 3
        [[0, 0, 0, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 1],
         [1, 1, 1, 0]]], dtype=np.int8)

    expected = np.array([
        [0.0, 0.6667931285555115, 0.5970146362973399, 1.0191352967734122],
        [0.5334914082114515, 0.0, 0.6989573069245543, 0.3937669813472373],
        [0.5970146362973399, 0.6989573069245543, 0.0, 0.42659206719560916],
        [0.9739191201245483, 0.3937669813472373, 0.23324008039305005, 0.0]],
        dtype=np.float64)

    ant_scores = np.array([0.8, 0.3, 0.2], dtype=np.float64)

    updateDirACS(paths=paths, P=P_t0, ant_scores=ant_scores, rho=evaporation, weight=1.5)

    assert np.all(np.round(P_t0, decimals=4) == np.round(expected, decimals=4)), \
        'FAILED TEST: antco.pheromone.updateDirACS()'

    print('SUCCESSFUL TEST: antco.pheromone.updateDirACS()')


def test():
    test_directed_AS_update()
    test_undirected_AS_update()
    test_directed_AS_elite_update()
    test_undirected_AS_elite_update()
    test_directed_MMAS_update()
    test_directed_MMAS_elite_update()
    test_undirected_MMAS_update()
    test_undirected_MMAS_elite_update()
    test_undirected_local_update()
    test_directed_local_update()
    test_undirected_ACS()
    test_directed_ACS()
