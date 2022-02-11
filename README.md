# Cannings

Tools for understanding a Cannings model for the evolution of a population.

## Cannings model used

In the module p0 is the probability that an individual has no offspring and alpha is such that the probability that an individual has more than k offpsprings is (1-p0) 1/(k^alpha).
The surviving offspring are then sampled so that there only N offsprings that survived where N is the size of the population. If there is less offspring than the population size, offspring are generated using a Wright-Fisher reproduction so that the number of offspring reach exactly the population size.

The fecundity selection allow the individuals of type A to have more offspring.
The viability selection allow the offspring of individuals of type A to have more chances to survive.

## Usage

The submodule cannings contains tools to modelise a Cannings reproduction with selection.
The file 'offspring_distribution' allows to compute a random draw of the number of offsprings of an individual.
The file 'diffusion' allows to compute the diffusion on an allele A tha have a selective advantage.
The file 'fixation' allows to compute the time to fixation or extinction of an allele A that have a selective advantage.

The submodule cannings_data allows to compute a lot of fixation times, to store the simulations and handle to data collected.
It is usefull to compare fecundity and viability selection. Yet it is not designed to have both fecundity and viability selection.

## Installation

`python setup.py install`

The following python modules have to be installed:
- numpy
- math
- scipy.special
- scipy.stats
