import numpy as np
from antco import stochasticAS


def test_stochastic_policy():
    """ antco.policy.stochastic_policy() unit testing """
    np.random.seed(1997)

    # Test if function works
    movements = np.array([1, 2, 3, 4])
    H = np.random.uniform(size=(5, 5)).astype(np.float64)
    P = np.random.uniform(size=(5, 5)).astype(np.float64)
    alpha = 0.5
    _ = stochasticAS(0, movements, H, P, alpha)

    # Test function logic (movement should be from 0 to 2)
    H[:, :] = 0.0
    P[:, :] = 0.0
    H[0, 2] = 10.0
    P[0, 2] = 10.0

    probs = stochasticAS(0, movements, H, P, alpha)

    assert np.argmax(probs) == 1, 'FAILED TEST: antco.policy.stochastic_policy()'

    print('SUCCESSFUL TEST: antco.policy.stochastic_policy()')


def test():
    test_stochastic_policy()
