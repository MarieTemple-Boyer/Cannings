""" Tools for the offspring distribution in a Cannings model where
    - the probability that an individidual has no offspring is p0
        (0 <= p0 <= 1)
    - the probability that an individual has more than k offpsrings (k>=1) is (1-p0) / (k**alpha)
        (0 < alpha)
"""

import numpy as np
from scipy.special import zeta
import math

def average(alpha, p0):
    """ Compute the average of the number of offspring of one individual for a Cannings model with parameters alpha and p0.

    >>> average(alpha=2, p0=0)
    1.6449340668482264
    >>> average(alpha=1.1, p0 = 0.5)
    5.292224232475402
    >>> average(alpha=1, p0=0)
    inf
    """
    
    check_parameters(alpha, p0)

    if alpha > 1:
        return (1-p0)*zeta(alpha)

    return math.inf

def check_parameters(alpha, p0):
    """ Check that the parameters alpha and p0 are admissible
        - 0 < alpha
        - 0 <= p0 <= 1
    >>> check_parameters(alpha=1, p0=0)

    >>> check_parameters(alpha=-1, p0=0)
    Traceback (most recent call last):
        ...
    Exception: alpha=-1 but it must respect 0 < alpha
    """
    if 0 > p0 or p0 > 1:
        raise Exception(f"p0={p0} but it must respect 0 <= p0 <= 1")
    if alpha <= 0:
        raise Exception(f"alpha={alpha} but it must respect 0 < alpha")


def generate_offspring(alpha, p0, nb_individuals=1):
    """ Generate randomly the number of offspring of nb_individuals individuals
    in the Cannings model with parameters alpha and p0

    >>> generate_offspring(alpha=1.4, p0=1)
    0
    >>> generate_offspring(alpha=2, p0=0) == 0
    False
    >>> np.random.seed(2)
    >>> generate_offspring(alpha=1, p0=0)
    2
    >>> generate_offspring(alpha=2, p0=0.1, nb_individuals=50)
    66
    """

    check_parameters(alpha, p0)

    if p0 == 1:
        return 0

    u = np.random.uniform(0, 1, nb_individuals)
    nb_offspring = (np.exp(1/alpha * np.log((1-p0)/u))).astype(int)

    return nb_offspring.sum()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
