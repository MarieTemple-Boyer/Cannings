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
    """

    nb_offsprings_type_A = np.round(
        (1+selection_coeff)*generate_offsprings(alpha, p0, nb_individuals_type_A)).astype(int)
    nb_other_offsprings = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)
    
    surviving_offsprings_type_A = np.random.hypergeometric(nb_offsprings_type_A, nb_other_offsprings, pop_size)

    return surviving_offsprings_type_A


def nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_viability(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    56
    """

    nb_offsprings_type_A = generate_offsprings(
        alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)

    ratio = (1 + selection_coeff) / \
        (1 + nb_individuals_type_A/pop_size * selection_coeff)
   
    nb_offsprings_type_A_eq = ratio*nb_offsprings_type_A
    nb_other_offsprings_eq = max(0, nb_other_offsprings + (1-ratio)*nb_offsprings_type_A)

    surviving_offsprings_type_A = np.random.hypergeometric(nb_offsprings_type_A_eq, nb_other_offsprings_eq, pop_size)

    return surviving_offsprings_type_A


def nb_next_generation(selection_type, nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a selective advantage (in a Cannings model)
    - selection_type : type of the selection. It can be either 'viability' or 'fecundity' (or 'fertility')
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation(selection_type='fertility', nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    63
    >>> nb_next_generation(selection_type='this_type_does_not_exist', nb_individuals_type_A=50, pop_size=100, alpha=2)
    Traceback (most recent call last):
        ...
    Exception: The selection type selection_type was 'this_type_does_not_exist' but it has to be either 'fecundity' (or 'fertility') or 'viability'.
    """
    if selection_type == 'fertility' or selection_type == 'fecundity':
        return nb_next_generation_fertility(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability':
        return nb_next_generation_viability(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    else:
        raise Exception(
            f"The selection type selection_type was '{selection_type}' but it has to be either 'fecundity' (or 'fertility') or 'viability'.")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
