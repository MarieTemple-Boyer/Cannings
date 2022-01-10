""" We consider a population of size pop_size with nb_offsprings_type_A individuals that have
the allele A and the other have the allele a in a Cannings model of parameters alpha and p0.
The allele A has a selective advantage selection_coeff.
The functions aim at computing the random number of individual of type A in the next generation
knowing the number of them in the current generation."""

import numpy as np

from cannings import generate_offsprings

def nb_next_generation_fertility(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a fertility selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    
    >>> np.random.seed(0)
    >>> nb_next_generation_fertility(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    62
    """

    nb_offsprings_type_A = np.round((1+selection_coeff)*generate_offsprings(alpha, p0, nb_individuals_type_A)).astype(int)
    nb_other_offsprings = generate_offsprings(alpha, p0, pop_size-nb_individuals_type_A)

    offsprings = np.concatenate((np.ones(nb_offsprings_type_A), np.zeros(nb_other_offsprings))).astype(int)
    np.random.shuffle(offsprings)
    surviving_offsprings_type_A = offsprings[0:pop_size].sum()

    return surviving_offsprings_type_A


def nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability exponential selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    
    >>> np.random.seed(1)
    >>> nb_next_generation_viability(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    77
    """

    nb_offsprings_type_A = generate_offsprings(alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(alpha, p0, pop_size-nb_individuals_type_A)

    offsprings = np.concatenate(
        (np.ones(nb_offsprings_type_A), np.zeros(nb_other_offsprings)))

    exp_realisation_A = np.random.exponential(
        1/(1+selection_coeff), size=nb_offsprings_type_A)
    exp_realisation_other = np.random.exponential(1, size=nb_other_offsprings)

    exp_realisation = np.concatenate(
        (exp_realisation_A, exp_realisation_other))

    order = np.argsort(exp_realisation)[0:pop_size]
    surviving_offprings_type_A = offsprings[order].sum()

    return surviving_offprings_type_A.astype(int)
    

def nb_next_generation_viability_gaussian(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0, variance=1):
    """ Compute the number of individuals of type A in the next generation with a viability exponential selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size: number total of individuals
    - alpha, p0: parameters for the Cannings model

    /!\ Issue: There may be less thant 0 or more than pop_size individuals of type A in the next generation (for instance if the variance is big)

    >>> np.random.seed(2)
    >>> nb_next_generation_viability_gaussian(nb_individuals_type_A=50, pop_size=100, alpha=1, selection_coeff=1)
    53
    """
    
    nb_offsprings_type_A = generate_offsprings(alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(alpha, p0, pop_size-nb_individuals_type_A)
    
    average = pop_size * (1+selection_coeff)*nb_offsprings_type_A / ((1+selection_coeff)*nb_offsprings_type_A + nb_other_offsprings)
     
    surviving_offprings_type_A = int(np.random.normal(average, variance))

    return surviving_offprings_type_A


if __name__ == "__main__":
    import doctest
    doctest.testmod()
