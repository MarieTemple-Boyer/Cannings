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
    
    def __init__(self, alpha, p0):
        
        """ Construc a Cannings model with parameters (alpha, p0)
        >>> sch = Schweinsberg(alpha=2, p0=0.2)
        >>> sch = Schweinsberg(alpha=1.1, p0=2)
        Traceback (most recent call last):
            ...
        Exception: p0=2 but it must respect 0 <= p0 <= 1
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

    def average(self):
        """ Compute the average of the number of offspring per individual.
        
        >>> sch = Schweinsberg(alpha=2, p0=0)
        >>> sch.average()
        1.6449340668482264
        """

        if self.alpha > 1:
            return (1-self.p0)*zeta(self.alpha)

        return math.inf

    def nb_next_generation(self, nb_individuals_type_1, pop_size, selection_fecundity=0, selection_viability=0, return_offspring_shortage=False, check_expectation=True):
        """ Return the number of individual of type 1 after one generation knowing that their
        number was nb_individuals_type_1 at the previous generation.
        The type 1 can have fecundity and viability selective advantage.
        If return_offspring_shortage then the function also return the shortage of offspring after
        the Cannings reproduction (that have been artificially added with a Wright-Fisher
        reproduction).
        If check_expectation the function return an Exception if the expectation of the number
        of offspring per individual is smaller than 1
        
        (see the documentation of Cannings.nb_next_generation for more details)
        
        >>> np.random.seed(0)
        >>> sch.nb_next_generation(nb_individuals_type_1=10, pop_size=100, selection_viability=2)
        6
        """
        if check_expectation:
            self.check_expectation()
        return Cannings.nb_next_generation(self, nb_individuals_type_1, pop_size, selection_fecundity=selection_fecundity, selection_viability=selection_viability, return_offspring_shortage=return_offspring_shortage)
        

    def fixation(self, pop_size, initial_nb_indiv_type_1=1, selection_fecundity=0, selection_viability=0, return_offspring_shortage=False, check_expectation=True):
        """ Compute the time to fixation or extinction of the type 1.
        It returns a couple (fixation, time).
            - fixation is True is the type 1 reached fixation and
              False the type 1 reached extinction
            - time is the time to fixation or extinction
        The type 1 can have fecundity and viability selective advantage.
        If return_offspring_shortage then the function also return the shortage of offspring
        after the Cannings reproduction as a list of tuple (generation, shortage).
            - generation is the generation where there have been a shortage
            - shortage is the number of offspring that have been artificially added with a
              Wright-Fisher reproduction
        If check_expectation the function return an Exception if the expectation of the number
        of offspring per individual is smaller than 1
        
        (see the documentation of Cannings.fixation for more details)
        
        >>> np.random.seed(0)
        >>> sch.fixation(pop_size=10, selection_fecundity=1)
        (False, 6)
        """

        if check_expectation:
            self.check_expectation()

        return Cannings.fixation(self, pop_size, initial_nb_indiv_type_1=initial_nb_indiv_type_1, selection_fecundity=selection_fecundity, selection_viability=selection_viability, return_offspring_shortage=return_offspring_shortage)

    def check_expectation(self):
        avg = self.average()
        if avg <= 1:
            raise Exception(
                f'The expectation of the number of offspring per individual is {avg  } but it should be greater than one.\nThe Cannings reproduction is hence not well defin  ed.')


def check_parameters(alpha, p0):
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
    sch.average()
    sch.nb_next_generation(10, 100)
    sch.fixation(100)

    import doctest
    doctest.testmod()
