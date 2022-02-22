# Cannings

Tools for understanding a Cannings model for the evolution of a population.


## Cannings model used

Each individual has offspring according to a distribution choosen by the user.
The surviving offspring are then sampled so that there only N offsprings that survived where N is the size of the population. If there is less offspring than the population size, offspring are generated using a Wright-Fisher reproduction so that the number of offspring reach exactly the population size.

The fecundity selection allow the individuals of type 1 to have more offspring.
The viability selection allow the offspring of individuals of type 1 to have more chances to survive.

## Usage

The submodule cannings contains the definition of the class Cannings that represent a Cannings reproduction.
It also contains the definition of two child classes of Cannings (Schweinsberg approximate a beta-coalescent and Poisson use a Poisson distribution).

The submodule schweinsberg_data allows to compute a lot of fixation times for a Schweinsberg reproduction, to store the simulations and handle to data collected.
It is usefull to compare fecundity and viability selection. Yet it is not designed to have both fecundity and viability selection.

## Installation

`python setup.py install`

The module is made for python 3.

The following python modules have to be installed:
- numpy
- math
- scipy.special
- scipy.stats
