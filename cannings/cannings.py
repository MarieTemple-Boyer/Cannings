""" Class for Cannings model

We consider a population of size N.
Each individual has a number of offspring drawn according to a certain distribution.
The N surviving offspring that will create the next generation are then drawn
among all the offspring.

Natural selection can be added to the model. The individual of type 1 will then have
a selective advantage (their are only two types possible).
For fecundity selection the number of offspring of the individuals of type 1 is
multiplied by the (1 + selection_fecundity)
For viability selection the surviving offspring are drawn according to the non central
Wallenius hypergeometric distribution with the offspring of type 1 having a weigth
(1 + selection_viability) and the other 1.
"""

import numpy as np
from scipy.stats import nchypergeom_wallenius


class Cannings:
    """ Class to implement a Cannings model from an offspring distribution"""

    def __init__(self, offspring_distribution, **parameters):
        """ Construct a Cannings model.
        offspring_distribution is a function that take in argument a number of individuals and
        eventually some parameters and return the number of offspring of those individuals.
        >>> def constant_distrib(nb_individuals, const):
        ...     return nb_individuals*const
        >>> constant = Cannings(constant_distrib, const=3)
        """
        self.offspring_distribution = offspring_distribution
        self.parameters = parameters

    def generate_offspring(self, nb_individuals=1):
        """ Return the number of offspring of nb_individuals
        >>> constant = Cannings(constant_distrib, const=3)
        >>> constant.generate_offspring()
        3
        >>> constant.generate_offspring(nb_individuals=5)
        15
        """
        return self.offspring_distribution(nb_individuals=nb_individuals, **self.parameters)

    def nb_next_generation(self, nb_individuals_type_1, pop_size,
                           selection_fecundity=0, selection_viability=0,
                           return_offspring_shortage=False):
        """ Return the number of individual of type 1 after one generation knowing that their
        number was nb_individuals_type_1 at the previous generation.
        The type 1 can have fecundity and viability selective advantage.
        If return_offspring_shortage then the function also return the shortage of offspring after
        the Cannings reproduction (that have been artificially added with a Wright-Fisher
        reproduction).
        >>> np.random.seed(0)
        >>> constant.nb_next_generation(nb_individuals_type_1=10, pop_size=100)
        12
        >>> constant.nb_next_generation(nb_individuals_type_1=10, pop_size=100,
        ...                             selection_viability=0.1, selection_fecundity=0.5)
        16
        >>> def poisson_distrib(nb_individuals, lanbda):
        ...     nb_offspring =  np.random.poisson(lanbda, size=nb_individuals).astype(int)
        ...     return nb_offspring.sum()
        >>> poisson = Cannings(poisson_distrib, lanbda=0.5)
        >>> poisson.nb_next_generation(nb_individuals_type_1=10, pop_size=20,
        ...                            return_offspring_shortage=True)
        (12, 4)
        >>> # There were 10 individual of type 1 out out 20 at first.
        >>> # At the next generations there are 12 individual of type 1
        >>> # but the reproduction of the Cannings model had a shortage of 4 offspring
        >>> # which where added artificially with a Wright-Fisher model.
        """
        nb_offspring_type_1 = int((1+selection_fecundity)*np.round(
            self.generate_offspring(nb_individuals=nb_individuals_type_1)))
        nb_other_offspring = self.generate_offspring(
            nb_individuals=pop_size-nb_individuals_type_1)

        nb_offspring_total = nb_offspring_type_1 + nb_other_offspring

        if nb_offspring_total >= pop_size:
            offspring_shortage = 0
            if selection_viability == 0:
                surviving_offspring_type_1 = np.random.hypergeometric(
                    nb_offspring_type_1, nb_other_offspring, pop_size)
            else:
                surviving_offspring_type_1 = nchypergeom_wallenius.rvs(
                    nb_offspring_total, nb_offspring_type_1, pop_size, 1+selection_viability)
        else:
            # there have not been enough offspring with the Cannings reproduction
            # we are adding offspring with a Wright-Fisher reproduction
            #   so that there are exactle pop_size offspring
            offspring_shortage = pop_size - nb_offspring_total
            surviving_offspring_type_1 = nb_offspring_type_1 + additional_offspring(
                nb_individuals_type_1, pop_size, pop_size-nb_offspring_total)

        if return_offspring_shortage:
            return surviving_offspring_type_1, offspring_shortage
        return surviving_offspring_type_1

    def fixation(self, pop_size, initial_nb_indiv_type_1=1,
                 selection_fecundity=0, selection_viability=0,
                 return_offspring_shortage=False):
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
        >>> np.random.seed(0)
        >>> np.random.seed(0)
        >>> constant.fixation(pop_size=100, selection_fecundity=1)
        (True, 15)
        >>> poisson.fixation(pop_size=10, selection_fecundity=1, return_offspring_shortage=True)
        (True, 6, [(1, 6), (2, 3), (3, 5)])
        >>> # The simulation began with one individual of type 1 in a population of size 10
        >>> # The type 1 has a fecundity selection advantage of parameter 1.
        >>> # The type 1 reach fixation after 6 generations.
        >>> # There was a shortage of offspring in the Cannings model at the generations 1, 2 and 3
        >>> # of respectively 6, 3 and 5 offspring.
        """
        assert 0 <= initial_nb_indiv_type_1 <= pop_size

        fixation = initial_nb_indiv_type_1 == pop_size
        extinction = initial_nb_indiv_type_1 == 0
        finished = fixation or extinction

        nb_indiv_type_1 = initial_nb_indiv_type_1
        nb_generations = 0
        offspring_shortage = []

        while not finished:
            nb_generations += 1

            res = self.nb_next_generation(nb_indiv_type_1, pop_size,
                                          selection_fecundity=selection_fecundity,
                                          selection_viability=selection_viability,
                                          return_offspring_shortage=return_offspring_shortage)

            if not return_offspring_shortage:
                nb_indiv_type_1 = res
            else:
                nb_indiv_type_1, shortage = res
                if shortage:
                    offspring_shortage.append((nb_generations, shortage))

            fixation = nb_indiv_type_1 == pop_size
            extinction = nb_indiv_type_1 == 0
            finished = fixation or extinction

        if return_offspring_shortage:
            return fixation, nb_generations, offspring_shortage
        return fixation, nb_generations


def additional_offspring(nb_individuals_type_1, pop_size, nb_missing_offspring):
    """ Compute the number of individual of type 1 after a  Wright-Fisher reproduction
        The reproducion generates exactly nb_missing_offspring.
        This function is used to complete the population if there are not enough offspring
        with the Cannings reproduction.
    >>> np.random.seed(0)
    >>> additional_offspring(50, 100, 3)
    2
    """

    nb_additional_offspring_type_1 = np.random.binomial(
        nb_missing_offspring, nb_individuals_type_1/pop_size)

    return nb_additional_offspring_type_1


if __name__ == "__main__":

    def constant_distrib(nb_individuals, const):
        return nb_individuals*int(const)

    constant = Cannings(constant_distrib, const=3)

    def poisson_distrib(nb_individuals, lanbda):
        nb_offspring = np.random.poisson(
            lanbda, size=nb_individuals).astype(int)
        return nb_offspring.sum()

    poisson = Cannings(poisson_distrib, lanbda=0.5)

    import doctest
    doctest.testmod()
