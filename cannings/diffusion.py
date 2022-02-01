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

    surviving_offsprings_type_A = np.random.hypergeometric(
        nb_offsprings_type_A, nb_other_offsprings, pop_size)

    return surviving_offsprings_type_A


def nb_next_generation_viability_hypergeometric(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_viability_hypergeometric(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    56
    """

    nb_offsprings_type_A = generate_offsprings(
        alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)

    ratio = (1 + selection_coeff) / \
        (1 + nb_individuals_type_A/pop_size * selection_coeff)

    nb_offsprings_type_A_eq = ratio*nb_offsprings_type_A
    nb_other_offsprings_eq = max(
        0, nb_other_offsprings + (1-ratio)*nb_offsprings_type_A)

    surviving_offsprings_type_A = np.random.hypergeometric(
        nb_offsprings_type_A_eq, nb_other_offsprings_eq, pop_size)

    return surviving_offsprings_type_A


def nb_next_generation_viability_exponential(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability exponential selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(1)
    >>> nb_next_generation_viability_exponential(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    77
    """

    nb_offsprings_type_A = generate_offsprings(
        alpha, p0, nb_individuals_type_A)
    nb_other_offsprings = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)

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


def nb_next_generation_viability_bernoulli(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability bernoulli selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation_viability_bernoulli(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    55
    """

    nb_offspring_type_A = generate_offsprings(
        alpha, p0, nb_individuals_type_A)
    nb_other_offspring = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)

    surviving_offspring = 0
    surviving_offspring_type_A = 0

    while surviving_offspring < pop_size:
        proba = nb_offspring_type_A / \
            (nb_offspring_type_A + nb_other_offspring)
        draw_offspring = np.random.binomial(1, proba)
        if draw_offspring:
            surviving_offspring += 1
            surviving_offspring_type_A += 1
            nb_offspring_type_A -= 1
        else:
            draw_selection = np.random.binomial(1, 1/(1+selection_coeff))
            if draw_selection:
                surviving_offspring += 1
                nb_other_offspring -= 1

    return surviving_offspring_type_A


def nb_next_generation_viability_bernoulli2(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability selection advantage (in a Cannings model)
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(1)
    >>> nb_next_generation_viability_bernoulli2(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    77
    """

    nb_offspring_type_A = generate_offsprings(
        alpha, p0, nb_individuals_type_A)
    nb_other_offspring = generate_offsprings(
        alpha, p0, pop_size-nb_individuals_type_A)

    def death_offspring(nb_offspring_type_A, nb_other_offspring):
        proba = (1+selection_coeff) / (2+selection_coeff)
        surviving_offspring_type_A = np.random.binomial(
            nb_offspring_type_A, proba)
        surviving_offspring_other = np.random.binomial(
            nb_other_offspring, 1-proba)
        return surviving_offspring_type_A, surviving_offspring_other

    surviving_offspring_type_A = 0
    surviving_offspring_other = 0

    while surviving_offspring_type_A+surviving_offspring_other < pop_size:
        surviving_offspring_type_A_this_time, surviving_offspring_other_this_time = death_offspring(
            nb_offspring_type_A, nb_other_offspring)
        nb_offspring_type_A -= surviving_offspring_type_A_this_time
        nb_other_offspring -= surviving_offspring_other_this_time
        surviving_offspring_type_A += surviving_offspring_type_A_this_time
        surviving_offspring_other += surviving_offspring_other_this_time

    truly_surviving_offspring_type_A = np.random.hypergeometric(
        surviving_offspring_type_A, surviving_offspring_other, pop_size)

    return truly_surviving_offspring_type_A


def nb_next_generation_viability(viability_type, nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0):
    """ Compute the number of individuals of type A in the next generation with a viability selective advantage (in a Cannings model)
    - viability_type :  type of viability considered (if the type of selection is 'viability'
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    """
    if viability_type == 'hypergeometric':
        return nb_next_generation_viability_hypergeometric(nb_individuals_type_A, pop_size, alpha, p0=p0, selection_coeff=selection_coeff)
    elif viability_type == 'exponential':
        return nb_next_generation_viability_exponential(nb_individuals_type_A, pop_size, alpha, p0=p0, selection_coeff=selection_coeff)
    elif viability_type == 'bernoulli':
        return nb_next_generation_viability_bernoulli(nb_individuals_type_A, pop_size, alpha, p0=p0, selection_coeff=selection_coeff)
    elif viability_type == 'bernoulli2':
        return nb_next_generation_viability_bernoulli2(nb_individuals_type_A, pop_size, alpha, p0=p0, selection_coeff=selection_coeff)
    else:
        raise Exception(
            f"The viability type was '{viability_type}' but it has to be 'hypergeometric', 'bernoulli' or 'exponential'.")


def nb_next_generation(nb_individuals_type_A, pop_size, alpha, p0=0, selection_coeff=0, selection_type='fertility'):
    """ Compute the number of individuals of type A in the next generation with a selective advantage (in a Cannings model)
    - selection_type : type of the selection. It can be either 'viability' or 'fecundity' (or 'fertility')
    - viability_type :  type of viability considered (if the type of selection is 'viability'
    - nb_individuals_type_A : number of individuals that have a selective advantage selection_coeff
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model

    >>> np.random.seed(0)
    >>> nb_next_generation(selection_type='viability_exponential', nb_individuals_type_A=50, pop_size=100, alpha=2, selection_coeff=1)
    52
    >>> nb_next_generation(nb_individuals_type_A=50, pop_size=100, alpha=2, selection_type = 'this_type_does_not_exist')
    Traceback (most recent call last):
        ...
    Exception: The selection type was 'this_type_does_not_exist' but it has to be one of those: 
    'fertility' (or 'fecundity'), 'viability', 'viability_hypergeometric', 'viability_exponential', 'viability_bernoulli'
    """
    if selection_type == 'fertility' or selection_type == 'fecundity':
        return nb_next_generation_fertility(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability' or selection_type == 'viability_hypergeometric':
        return nb_next_generation_viability_hypergeometric(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability_exponential':
        return nb_next_generation_viability_exponential(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    elif selection_type == 'viability_bernoulli':
        return nb_next_generation_viability_bernoulli(nb_individuals_type_A, pop_size, alpha, p0, selection_coeff)
    else:
        available_selection_type = "'fertility' (or 'fecundity'), 'viability', 'viability_hypergeometric', 'viability_exponential', 'viability_bernoulli'"
        raise Exception(
                f"The selection type was '{selection_type}' but it has to be one of those: \n" + available_selection_type)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
