"""
Set of class used to model the data stored in a json file thanks to the functions from 'generate_data.py'.
"""

import json
from tabulate import tabulate


class CompleteData:
    """ Class for the data of times and probability of fixations with the hyperparameters and parameters associated.
    >>> complete_data = CompleteData('data_example/permanent.json')
    """

    def __init__(self, file_name):
        with open(file_name, 'r') as read_file:
            complete_data = json.load(read_file)
        hyp = complete_data[0]
        self.hyperameters = Hyperparameters(
            hyp['model'], hyp['population_size'], hyp['p0'], hyp['selection'])
        all_data = complete_data[1]
        self.fixations = [FixationSet(
            data['alpha'], data['selection_coefficient'], data['fixation']) for data in all_data]

    def alpha(self, selection_coeff=None):
        """
        Return the values of alpha for which data are stored.
        If selection_coeff is not None the it returns the value of alpha for which data are stored with the choosen selection coefficient. 
        >>> complete_data.alpha()
        [1.1, 1.5]
        """

        if selection_coeff is None:
            available = [data.alpha for data in self.fixations]
        else:
            available = [data.alpha
                         for data in self.fixations if data.selection_coeff == selection_coeff]

        return sorted(list(set(available)))

    def selection_coeff(self, alpha=None):
        """
        Return the values of selection coefficinet for which data are stored.
        If alpha is not None the it returns the value of selection coefficient for which data are stored with the choosen alpha. 

        >>> complete_data.selection_coeff(alpha=1.5)
        [0.1]
        """

        if alpha is None:
            available = [data.selection_coeff for data in self.fixations]
        else:
            available = [data.selection_coeff
                         for data in self.fixations if data.alpha == alpha]

        return sorted(list(set(available)))

    def exists(self, alpha, selection_coeff):
        """
        Check if there is a fixation set whose parameters are alpha and selection_coeff.
        >>> complete_data.exists(1.1, 0.1)
        True
        >>> complete_data.exists(2, 0)
        False
        """
        return alpha in self.alpha(selection_coeff=selection_coeff)

    def fixation_set(self, alpha, selection_coeff):
        """
        Return the FixationSet associated with choosen alpha and selection coefficient.
        >>> print(complete_data.fixation_set(1.1, 0.1))
        alpha: 1.1 | selection coefficient: 0.1
        [-1, -1, -1, -5, -4, -1, -3, -1, -4, -2]
        >>> print(complete_data.fixation_set(2, 0)) # does not exists
        None
        """
        if self.exists(alpha, selection_coeff):
            return next(fixation_set for fixation_set in self.fixations if fixation_set.alpha == alpha and fixation_set.selection_coeff == selection_coeff)
        else:
            return None

    def print_nb_iterations(self):
        """ Print a table with two entries (alpha and the selection coefficient) giving the number of iterations for each set of values/
        >>> complete_data.print_nb_iterations()
        Number of iterations
        -----------------  ----  ----
        selection \ alpha   1.1   1.5
        0.1                10    10
        1                  10     0
        -----------------  ----  ----
        """
        alpha_values = ['selection \ alpha'] + self.alpha()
        tab_iter = [alpha_values]
        for selec in self.selection_coeff():
            this_line = [selec]
            for alpha in self.alpha():
                if self.exists(alpha, selec):
                    this_line.append(self.fixation_set(
                        alpha, selec).nb_iterations())
                else:
                    this_line.append(0)
            tab_iter.append(this_line)

        print('Number of iterations')
        print(tabulate(tab_iter))

    def print_nb_fixations(self):
        """ Print a table with two entries (alpha and the selection coefficient) giving the number of fixations for each set of values/
        >>> complete_data.print_nb_fixations()
        Number of fixations
        -----------------  ---  ---
        selection \ alpha  1.1  1.5
        0.1                0    2
        1                  3    0
        -----------------  ---  ---
        """
        alpha_values = ['selection \ alpha'] + self.alpha()
        tab_iter = [alpha_values]
        for selec in self.selection_coeff():
            this_line = [selec]
            for alpha in self.alpha():
                if self.exists(alpha, selec):
                    this_line.append(self.fixation_set(
                        alpha, selec).nb_fixations())
                else:
                    this_line.append(0)
            tab_iter.append(this_line)

        print('Number of fixations')
        print(tabulate(tab_iter))


class Hyperparameters:
    """ Class for the hyperparameters.
        model: model use for the reproduction (Cannings for instance)
        population_size: size of the population
        p0: probability that an individual has no offspring
        selection: type of selection (viability or fecundity for instance)
    """

    def __init__(self, model, population_size, p0, selection):
        self.model = model
        self.population_size = population_size
        self.p0 = p0
        self.selection = selection


class FixationSet:
    """ Class for a table of times of fixations or extinctions and the parameters associated.
    """

    def __init__(self, alpha, selection_coeff, times):
        self.alpha = alpha
        self.selection_coeff = selection_coeff
        self.times = times

    def nb_iterations(self):
        """ Return the number of simulations.
        >>> fix.nb_iterations()
        10
        """
        return len(self.times)

    def fixation_times(self):
        """ Return the fixations times.
        >>> fix.fixation_times()
        [10, 5, 13]
        """
        return [time for time in self.times if time > 0]

    def nb_fixations(self):
        """ Return the number of fixations.
        >>> fix.nb_fixations()
        3
        """
        return len(self.fixation_times())

    def proba_fixation(self):
        """ Return the probability of fixation.
        >>> fix.proba_fixation()
        0.3
        """
        return self.nb_fixations() / self.nb_iterations()

    def avg_fixation_time(self):
        """ Return the average of the fixation time (assuming there was a fixation.
        >>> fix.avg_fixation_time()
        9.333333333333334
        """
        if self.nb_fixations() == 0:
            return None
        return sum(self.fixation_times()) / self.nb_fixations()

    def __str__(self):
        return f'alpha: {self.alpha} | selection coefficient: {self.selection_coeff}' + '\n' + str(self.times)


if __name__ == '__main__':
    import doctest
    file_name = 'data_example/permanent.json'
    complete_data = CompleteData(file_name)
    fix = complete_data.fixation_set(1.1, 1)
    doctest.testmod()
