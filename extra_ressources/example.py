""" Template to start using the module 'cannings' """

import numpy as np
import cannings

# Using the class Schweinsberg

pop_size = 50
alpha = 1.6
p0 = 0.5

can = cannings.Schweinsberg(alpha, p0)

current_nb_type_1 = 10

# Average of the number of offspring per individual
avg = can.average()
print(f'One individual has one average {avg} offspring.')

# Random number of offspring of one individual
n_offspring = can.generate_offspring()
print(f'This time, one individuals had {n_offspring} offspring.')
# Random number of offspring of several individuals
n_indiv = 10
n_offspring = can.generate_offspring(nb_individuals=n_indiv)
print(f'That time, {n_indiv} individuals had {n_offspring} offspring.')

# Number of individuals of type A in the next generation knowing the current number of invidiual of type A
#   Without any selective advantage
new_nb_indiv_A = can.nb_next_generation(current_nb_type_1, pop_size)
print(
    f'At the current generation there are {current_nb_type_1} individuals of type A and at the next there will be {new_nb_indiv_A}.')
#   With fecundity selection
new_nb_indiv_1 = can.nb_next_generation(current_nb_type_1, pop_size, selection_fecundity=1)
#   With viability selection
new_nb_indiv_1 = can.nb_next_generation(current_nb_type_1, pop_size, selection_viability=1)
#   With both selection
new_nb_indiv_1 = can.nb_next_generation(current_nb_type_1, pop_size, selection_fecundity=0.5, selection_viability=0.5)

#   We can also know if some offspring have been artificially generated using a Wright-Fisher reprocution.
#   (if the case where there was not enough offspring with the Cannings reproduction)
new_nb_indiv_A, nb_indiv_artificially_generated = can.nb_next_generation(
    current_nb_type_1, pop_size, return_offspring_shortage=True)
print(f'There has been {nb_indiv_artificially_generated} individuals artificially generated using a Wright-Fisher reproduction in order to reach the population size {pop_size}.')


# Time to fixation or extinction of the allele A
#   Initially one individual has the type A
fixation, nb_generations = can.fixation(pop_size, selection_fecundity=0.5, selection_viability=1)
if fixation:
    print(
        f'This time the allele A is fixed after {nb_generations} generations')
else:
    print(
        f'That time the allele A is extincted after {nb_generations} generations')


# The functions automatically check that the expectation of the number of offspring per individual is greater than 1 and raise an error if not.
# This can be disabled using this:
new_nb_indiv_A = can.nb_next_generation(
    current_nb_type_1, pop_size, check_expectation=False)
fixation, nb_generations = can.fixation(
    pop_size, check_expectation=False)



# Using directly the class Cannings

def exponential_distrib(nb_individuals, param):
    value = np.random.exponential(param, size=nb_individuals)
    return value.sum()

can = cannings.Cannings(exponential_distrib, param=2.)

# We also have the functions presented precedently
avg = can.average()
new_nb_indiv_1 = can.nb_next_generation(current_nb_type_1, pop_size)
fixation, time = can.fixation(pop_size)
