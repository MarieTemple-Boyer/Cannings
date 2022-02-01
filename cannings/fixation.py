""" We consider a population of size pop_size in a Cannings model of parameters alpha and p0.
There are two alleles and the  allele A has a selective advantage selection_coeff.
knowing the number of them in the current generation.
The functions aim at computing the probability of fixation.
"""

import numpy as np
from cannings import nb_next_generation


def fixation(pop_size, alpha, p0=0, selection_coeff=0, initial_nb_indiv_A=1, selection_type='fertility'):
    """
    Compute the time to fixation of the allele A that have a selective advantage (in a Cannings model)
    - pop_size: size of the population
    - alpha, p0: parameters for the Cannings model
    - selection_coeff: fertility selection coefficient of the allele A
    - initial_nb_indiv_A: number of invididuals that have the allele A at the first generation
    - selection_type: type of the selection (either 'fertility' (or 'fecundity') or 'viability'). By default it is a fecundity selection

    It return a couple (fixation, nb_generation).
    - fixation (bool): is True if all the individual have the allele A at the end
    - nb_generations (integer): is fixation this is the time to fixation else this is the time to extinction

    >>> # there no individual of type A so their is an extinction at the generation 0
    >>> fixation(pop_size=100, alpha=1.1, initial_nb_indiv_A=0)
    (False, 0)
    >>> np.random.seed(2)
    >>> # the allele A is fixed at the generation 29
    >>> fixation(pop_size=100, alpha=2, selection_coeff=0.1, initial_nb_indiv_A=10)
    (True, 117)
    >>> fixation(pop_size=100, alpha=1.4, selection_coeff=1, selection_type='viability_exponential')
    (False, 6)
    """

    assert 0 <= initial_nb_indiv_A and initial_nb_indiv_A <= pop_size

    fixation = initial_nb_indiv_A == pop_size
    extinction = initial_nb_indiv_A == 0
    finished = fixation or extinction

    nb_indiv_A = initial_nb_indiv_A
    nb_generations = 0

    while not finished:
        nb_generations += 1
        nb_indiv_A = nb_next_generation(
            nb_indiv_A, pop_size, alpha, p0=p0, selection_coeff=selection_coeff, selection_type=selection_type)

        fixation = nb_indiv_A == pop_size
        extinction = nb_indiv_A == 0
        finished = fixation or extinction

    return fixation, nb_generations


if __name__ == "__main__":
    import doctest
    doctest.testmod()
