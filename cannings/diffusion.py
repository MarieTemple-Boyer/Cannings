""" We consider a population of size pop_size with nb_offsprings_type_A individuals that have
the allele A and the other have the allele a in a Cannings model of parameters alpha and p0.
The allele A has a selective advantage selection_coeff.
The functions aim at computing the random number of individual of type A in the next generation
knowing the number of them in the current generation."""

import numpy as np

from offsprings_distribution import generate_offsprings

def nb_next_generation_fertility(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a fertility selection advantage
    >>> np.random.seed(0)
    >>> nb_next_generation_fertility(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    62
    """

    nb_offsprings_type_A = (1+selection_coeff)*generate_offsprings(alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(alpha, p0, pop_size-nb_individuals_type_A)
    
    offsprings = np.concatenate((np.ones(nb_offsprings_type_A), np.zeros(nb_other_offsprings)))
    np.random.shuffle(offsprings)
    surviving_offsprings_type_A = offsprings[0:pop_size].sum()

    return surviving_offsprings_type_A.astype(int)



if __name__ == "__main__":
    import doctest
    doctest.testmod()
