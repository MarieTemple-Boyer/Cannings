""" Class for a Cannings model with an offspring distribution that approximates
a beta coalescent of parameter alpha
(Schweinsberg, 2003, 'Coalescent processes obtained from supercritical Galton–Watson processes')

The probability that an individual has no offspring is p0.
The propability that an individual has more than k offspring is (1-p0) / (k**alpha)
"""

import math
import numpy as np
from scipy.special import zeta
from scipy.stats import nchypergeom_wallenius

from cannings import Cannings


class Schweinsberg(Cannings):
    """ Class to implement a Cannings model that approximates a beta-coalescent"""

    def __init__(self,
                 alpha: float,
                 p0: float):
        """ Construct a Cannings model with parameters (alpha, p0)
        >>> sch = Schweinsberg(alpha=1.5, p0=0.1)
        >>> np.random.seed(0)
        >>> sch.generate_offspring(nb_individuals=3)
        3
        >>> sch.nb_next_generation(nb_individuals_type_1=10, pop_size=100)
        5
        >>> sch.fixation(pop_size=100, selection_viability=1)
        (True, 11)
        """
        check_parameters(alpha, p0)

        def distrib(nb_individuals, alpha, p0):
            if p0 == 1:
                return 0
            random_uni = np.random.uniform(0, 1, nb_individuals)
            nb_offspring = (
                np.exp(1/alpha * np.log((1-p0)/random_uni))).astype(int)
            return nb_offspring.sum()

        Cannings.__init__(self, distrib, alpha=alpha, p0=p0)
        self.alpha = alpha
        self.p0 = p0

    def average(self) -> float:
        """ Compute the average of the number of offspring per individual.

        >>> sch = Schweinsberg(alpha=2, p0=0)
        >>> sch.average()
        1.6449340668482264
        """

        if self.alpha > 1:
            return (1-self.p0)*zeta(self.alpha)

        return math.inf


def check_parameters(alpha: float, p0: float) -> None:
    """ Check that the parameters alpha and p0 are admissible
        - 0 < alpha
        - 0 <= p0 <= 1
    """
    if 0 > p0 or p0 > 1:
        raise Exception(f"p0={p0} but it must respect 0 <= p0 <= 1")
    if alpha <= 0:
        raise Exception(f"alpha={alpha} but it must respect 0 < alpha")


if __name__ == "__main__":

    sch = Schweinsberg(alpha=1.5, p0=0.1)

    import doctest
    doctest.testmod()
