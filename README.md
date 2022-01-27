# Cannings

Tools for understanding a Cannings model for the evolution of a population.

## Cannings model used

In the module p0 is the probability that an individual has no offsprings and alpha is such that the probability that an individual has more than k offpsprings is (1-p0) 1/(k^alpha).
The surviving offsprings are then sampled so that there only pop_size offsprings that survived.

The fecundity (or fertility) selection allow the individuals of type A to have more offsprings.
The viability selection allow the offspring of individuals of type A to have more chances to survive.

## Usage

The submodule offsprings_distribution allows to compute a random draw of the number of offsprings of an individual.

The submodule diffusion allows to compute the diffusion on an allele A tha have a selective advantage.

The submodule fixation allows to compute the time to fixation or extinction of an allele A that have a selective advantage.

## Installation

`python setup.py install`

The following python modules have to be installed:
- numpy
- math
- scipy.special
