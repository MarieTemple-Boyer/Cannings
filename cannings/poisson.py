""" Class for a Cannings model with an Poisson offspring distribution.
When the average of the distribution is greater than one but close to one
this should approximate a Wright-Fisher model.
"""

import numpy as np

from cannings import Cannings


class Poisson(Cannings):
    """ Class to implement a Cannings model with a Poisson distribution"""

    def __init__(self, lanbda):
        """ Construct a Cannings model with a Poisson distribution of parameter lanbda
        >>> pois = Poisson(lanbda=1.1)
        >>> np.random.seed(0)
        >>> pois.generate_offspring(nb_individuals=4)
        6
        >>> pois.nb_next_generation(nb_individuals_type_1=10, pop_size=100)
        14
        >>> pois.fixation(pop_size=100, selection_fecundity=1)
        (False, 1)
        """

        def distrib(nb_individuals, lanbda):
            nb_offspring = np.random.poisson(lanbda, size=nb_individuals)
            return nb_offspring.sum()

        Cannings.__init__(self, distrib, lanbda=lanbda)
        self.lanbda = lanbda

    def average(self):
        """ Compute the average of the number of offspring per individual.

        >>> pois.average()
        1.1
        """

        return self.lanbda


if __name__ == "__main__":

    pois = Poisson(lanbda=1.1)

    import doctest
    doctest.testmod()
