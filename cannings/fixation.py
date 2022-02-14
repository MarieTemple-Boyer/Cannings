""" We consider a population of size pop_size in a Cannings model of parameters alpha and p0.
There are two alleles and the  allele A has a selective advantage selection_coeff.
knowing the number of them in the current generation.
The functions aim at computing the probability of fixation.
"""

import numpy as np
from cannings import nb_next_generation


def fixation(pop_size, alpha, p0=0, initial_nb_indiv_A=1, selection_fecundity=0, selection_viability=0, check_expectation=True, return_offspring_shortage=False):
    """
    Compute the time to fixation of the allele A that have a selective advantage (in a Cannings model)
    It return a couple (fixation, nb_generation).
    - fixation (bool): is True if all the individual have the allele A at the end
    - nb_generations (integer): is fixation this is the time to fixation else this is the time to extinction

    
    - pop_size : number total of individuals
    - alpha, p0 : parameters for the Cannings model
    - initial_nb_indiv_A: number of invididuals that have the allele A at the first generation
    - selection_fecundity : coefficient for the fecundity selection (if 0 there is no fecundity selection)
    - selection_viability : coefficient for the viability selection (if 0 there is no viability selection)
    - check_expectation : if True then raise an exception if the expectation of the numbers of offspring per individul is smaller than selection_fecundity
    - return_offsprinf_shortage : if True then the function return an additional value that is a list of couple.
        The first value is the generation.
        The second is such that:
            If there are more offspring than the population size with the Cannings reproduction in the generation considered then this number is always 0.
            Else this is the number of offspring generated with a Wright-Fisher model to reach the size of the population.

    >>> # there no individual of type A so their is an extinction at the generation 0
    >>> fixation(pop_size=100, alpha=1.1, initial_nb_indiv_A=0)
    (False, 0)
    >>> np.random.seed(0)
    >>> # the allele A is fixed at the generation 29
    >>> fixation(pop_size=100, alpha=2, initial_nb_indiv_A=10, selection_fecundity=0.1, selection_viability=1)
    (True, 15)
    >>> fixation(pop_size=100, alpha=2, initial_nb_indiv_A=10, p0=0.3, return_offspring_shortage=True)
    (False, 7, [(3, 8), (6, 2)])
    >>> #(it took 7 generation to reach the extinction and at the generations 3 and 6 there were
    >>> # respectively 8 and 2 offspring generated with a WF reproduction)
    """

    assert 0 <= initial_nb_indiv_A and initial_nb_indiv_A <= pop_size

    fixation = initial_nb_indiv_A == pop_size
    extinction = initial_nb_indiv_A == 0
    finished = fixation or extinction

    nb_indiv_A = initial_nb_indiv_A
    nb_generations = 0
    offspring_shortage = []

    while not finished:
        nb_generations += 1

        if not return_offspring_shortage:
            nb_indiv_A = nb_next_generation(
                nb_indiv_A, pop_size, alpha, p0=p0,
                selection_fecundity=selection_fecundity, selection_viability=selection_viability)
        
        else:
            nb_indiv_A, shortage = nb_next_generation(
                nb_indiv_A, pop_size, alpha, p0=p0,
                selection_fecundity=selection_fecundity, selection_viability=selection_viability,
                return_offspring_shortage=True)
            
            if shortage:
                offspring_shortage.append((nb_generations, shortage))
         
        fixation = nb_indiv_A == pop_size
        extinction = nb_indiv_A == 0
        finished = fixation or extinction
    
    if return_offspring_shortage:
        return fixation, nb_generations, offspring_shortage
    return fixation, nb_generations


if __name__ == "__main__":
    import doctest
    doctest.testmod()
