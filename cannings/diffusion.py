""" We consider a population of size pop_size with nb_offspring_type_A individuals that have
the allele A and the other have the allele a in a Cannings model of parameters alpha and p0.
The allele A has a selective advantage selection_coeff.
The functions aim at computing the random number of individual of type A in the next generation
knowing the number of them in the current generation."""

import numpy as np

from cannings import generate_offspring, average
from scipy.stats import nchypergeom_wallenius


def additional_offspring(nb_individuals_type_A, pop_size, nb_missing_offspring):
    """ Compute an additional number of offspring of type A if there are not enough offspring. The additional offspring are generated using a Wright-Fisher model.
    - nb_individuals_type_A : number of individuals (parents) that have a selective advantage selection_coeff
    - pop_size = number total of individuals (parents)
    - nb_missing_offspring : number of offspring to generate using a Wright-Fisher model
        (for instance if the number of offspring after a Cannings reproduction is smaller than the population size)

    >>> np.random.seed(0)
    >>> additional_offspring(50, 100, 3)
    2
    """

    nb_additional_offspring_type_A = np.random.binomial(
        nb_missing_offspring, nb_individuals_type_A/pop_size)

    return nb_additional_offspring_type_A


def nb_next_generation_neutral(nb_individuals_type_A, pop_size, alpha, p0=0):
    """ Compute the number of individuals of type A in the next generation with a fecundity selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_neutral(nb_individuals_type_A=50, pop_size=100, alpha=2)
    46
    """
    nb_offspring_type_A = np.round(generate_offspring(
        alpha, p0, nb_individuals_type_A)).astype(int)
    nb_other_offspring = generate_offspring(
        alpha, p0, pop_size-nb_individuals_type_A)

    nb_offspring_total = nb_offspring_type_A + nb_other_offspring

    if nb_offspring_total >= pop_size:
        surviving_offspring_type_A = np.random.hypergeometric(
            nb_offspring_type_A, nb_other_offspring, pop_size)
    else:
        surviving_offspring_type_A = nb_offspring_type_A + additional_offspring(
            nb_individuals_type_A, pop_size, pop_size-nb_offspring_total)

    return surviving_offspring_type_A


def nb_next_generation_fecundity(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a fecundity selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_fecundity(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    63
    """

    nb_offspring_type_A = np.round(
        (1+selection_coeff)*generate_offspring(alpha, p0, nb_individuals_type_A)).astype(int)
    nb_other_offspring = generate_offspring(
        alpha, p0, pop_size-nb_individuals_type_A)

    nb_offspring_total = nb_offspring_type_A + nb_other_offspring

    if nb_offspring_total >= pop_size:
        surviving_offspring_type_A = np.random.hypergeometric(
            nb_offspring_type_A, nb_other_offspring, pop_size)
    else:
        surviving_offspring_type_A = nb_offspring_type_A + additional_offspring(
            nb_individuals_type_A, pop_size, pop_size-nb_offspring_total)

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

    nb_offspring_total = nb_offspring_type_A + nb_other_offspring

    if nb_offspring_total >= pop_size:
        surviving_offspring_type_A = nchypergeom_wallenius.rvs(
            nb_offspring_total, nb_offspring_type_A, pop_size, 1+selection_coeff)
    else:
        return nb_offspring_type_A + additional_offspring(nb_individuals_type_A, pop_size, pop_size-nb_offspring_total)

    return surviving_offspring_type_A


def nb_next_generation(nb_individuals_type_A, pop_size, alpha, p0=0, selection_fecundity=0, selection_viability=0, check_expectation=True):
    """
    >>> np.random.seed(0) 
    >>> nb_next_generation(nb_individuals_type_A=10, pop_size=100, alpha=1.3, selection_viability=1)
    7
    >>> nb_next_generation(nb_individuals_type_A=10, pop_size=100, alpha=2, p0=0.5, selection_fecundity=1)
    Traceback (most recent call last):
        ...
    Exception: The expectation of the number of offspring per individual is 0.8224670334241132 but it should be greater than one.
    The Cannings reproduction is hence not well defined.
    """

    if check_expectation:
        exp = average(alpha, p0)
        if exp <= 1:
            raise Exception(
                f'The expectation of the number of offspring per individual is {exp} but it should be greater than one.\nThe Cannings reproduction is hence not well defined.')

    nb_offspring_type_A = int((1+selection_fecundity)*np.round(generate_offspring(
        alpha, p0, nb_individuals_type_A)))
    nb_other_offspring = generate_offspring(
        alpha, p0, pop_size-nb_individuals_type_A)

    nb_offspring_total = nb_offspring_type_A + nb_other_offspring

    if nb_offspring_total >= pop_size:
        if selection_viability == 0:
            surviving_offspring_type_A = np.random.hypergeometric(
                nb_offspring_type_A, nb_other_offspring, pop_size)
        else:
            surviving_offspring_type_A = nchypergeom_wallenius.rvs(
                nb_offspring_total, nb_offspring_type_A, pop_size, 1+selection_viability)
    else:
        # there have not been enough offspring with the Cannings reproduction
        # we are adding offspring with a Wright-Fisher reproduction
        #   so that there are exactle pop_size offspring
        surviving_offspring_type_A = nb_offspring_type_A + additional_offspring(
            nb_individuals_type_A, pop_size, pop_size-nb_offspring_total)

    return surviving_offspring_type_A


def nb_next_generation1(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0, selection_type='neutral', check_expectation=True):
    """ Compute the number of individuals of type A in the next generation with a selective advantage (in a Cannings model)
    - selection_type : type of the selection. It can be either 'viability', 'fecundity' or 'neutral' if their is no selection
    - selection_coeff : coefficient of selection if the selection type is not 'neutral'
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    - check_expectation : if True then raise an exception if the expectation of the numbers of offspring per individul is smaller than one

    >>> np.random.seed(0)
    >>> nb_next_generation1(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1, selection_type='viability')
    52
    >>> nb_next_generation1(nb_individuals_type_A=50, pop_size=100, alpha=2, p0=0.1)
    41
    >>> nb_next_generation1(nb_individuals_type_A=50, pop_size=100, alpha=2, p0=0.5)
    Traceback (most recent call last):
        ...
    Exception: The expectation of the number of offspring per individual is 0.8224670334241132 but it should be greater than one.
    The Cannings reproduction is hence not well defined.
    >>> nb_next_generation1(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_type = 'this_type_does_not_exist')
    Traceback (most recent call last):
        ...
    Exception: The selection type was 'this_type_does_not_exist' but it has to be one of those: 'viability', 'fecundity' or 'neutral'.
    """

    if check_expectation:
        exp = average(alpha, p0)
        if exp <= 1:
            raise Exception(
                f'The expectation of the number of offspring per individual is {exp} but it should be greater than one.\nThe Cannings reproduction is hence not well defined.')

    if selection_type == 'fecundity':
        return nb_next_generation_fecundity(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability':
        return nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'neutral':
        return nb_next_generation_neutral(nb_individuals_type_A, pop_size, alpha, p0)
    else:
        raise Exception(
            f"The selection type was '{selection_type}' but it has to be one of those: 'viability', 'fecundity' or 'neutral'.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
