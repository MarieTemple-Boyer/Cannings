""" We consider a population of size pop_size with nb_offspring_type_A individuals that have
the allele A and the other have the allele a in a Cannings model of parameters alpha and p0.
The allele A has a selective advantage selection_coeff.
The functions aim at computing the random number of individual of type A in the next generation
knowing the number of them in the current generation."""

import numpy as np

from cannings import generate_offspring
from scipy.stats import nchypergeom_wallenius


def nb_next_generation_fecundity(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a fecundity selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    """

    nb_offspring_type_A = np.round(
        (1+selection_coeff)*generate_offspring(alpha, p0, nb_individuals_type_A)).astype(int)
    nb_other_offspring = generate_offspring(
        alpha, p0, pop_size-nb_individuals_type_A)

    surviving_offspring_type_A = np.random.hypergeometric(
        nb_offspring_type_A, nb_other_offspring, pop_size)

    return surviving_offspring_type_A


def nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability selection advantage (in a Cannings model).
    An non neutral Wallenius hypergeometric distribution is considered.
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_viability(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    52
    """

    nb_offspring_type_A = generate_offspring(
        alpha, p0, nb_individuals_type_A)
    nb_other_offspring = generate_offspring(
        alpha, p0, pop_size-nb_individuals_type_A)
    nb_total_offspring = nb_offspring_type_A + nb_other_offspring

    surviving_offspring_type_A = nchypergeom_wallenius.rvs(
        nb_total_offspring, nb_offspring_type_A, pop_size, 1+selection_coeff)

    return surviving_offspring_type_A


def nb_next_generation(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0, selection_type='fecundity'):
    """ Compute the number of individuals of type A in the next generation with a selective advantage (in a Cannings model)
    - selection_type : type of the selection. It can be either 'viability' or 'fecundity'
    - viability_type :  type of viability considered (if the type of selection is 'viability'
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation(selection_type='viability', nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    52
    >>> nb_next_generation(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_type = 'this_type_does_not_exist')
    Traceback (most recent call last):
        ...
    Exception: The selection type was 'this_type_does_not_exist' but it has to be one of those 'viability' or 'fecundity.
    """
    if selection_type == 'fecundity':
        return nb_next_generation_fecundity(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability':
        return nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    else:
        raise Exception(
            f"The selection type was '{selection_type}' but it has to be one of those 'viability' or 'fecundity.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
