""" Set of usefull functions to compute the probability and the time to fixation
and store them in a json file.

Architecture of the json file:

[ hyper_parameters, all_data ]

    -hyper_parameters: {'model':(string), 'population_size':(int), 'p0':(float), selection:(string)}
    -all_data: [data1, data2, ..., datan]

        -data: {'alpha':(float), 'selection_coefficient':(float),
                'fixation':[time1, time2, ..., timem] }

            -time:(int) if time > 0 then this is the time to fixation
                                    else this is the time to extinction
"""

from os.path import exists
from os import remove
import json
from cannings import Schweinsberg


def initialise_file(file_name,
                    pop_size,
                    p0,
                    selection_type):
    """ Initialise the hyper parameters (pop_size, p0, selection) in  the json file where the data
    will be stored."""

    hyper_parameters = {'model': 'cannings',
                        'population_size': pop_size, 'p0': p0, 'selection': selection_type}
    initial_data = [hyper_parameters, []]

    with open(file_name, 'w') as write_file:
        json.dump(initial_data, write_file)


def check_hyper_parameters(file_name,
                           model,
                           pop_size,
                           p0,
                           selection_type):
    """ Check that the hyper parmeters (pop_size, p0, selection) in argument are the same as the one
    in file_name.

    >>> file_name = 'data_example/permanent.json'
    >>> check_hyper_parameters(file_name, 'cannings', 100, 0.1, 'fecundity')
    True
    """

    with open(file_name, 'r') as read_file:
        complete_data = json.load(read_file)
        hyper_parameters = complete_data[0]
    everything_fine = (hyper_parameters['model'] == model
                       and hyper_parameters['population_size'] == pop_size
                       and hyper_parameters['p0'] == p0
                       and hyper_parameters['selection'] == selection_type)
    return everything_fine


def initialise_data(file_name,
                    pop_size,
                    alpha,
                    p0,
                    selection_coeff,
                    selection_type,
                    model='cannings'):
    """ Initialise the parameters (alpha, selection_coeff) in the json file where the data
    will be stored."""

    if not exists(file_name):
        initialise_file(file_name, pop_size, p0, selection_type)

    everything_fine = check_hyper_parameters(
        file_name, model, pop_size, p0, selection_type)
    if not everything_fine:
        raise Exception(
            f'The file {file_name} has not been the right hyperameters.')

    with open(file_name, 'r') as read_file:
        data_complete = json.load(read_file)
        all_data = data_complete[1]

    stored_data = [data for data in all_data if data['alpha'] ==
                   alpha and data['selection_coefficient'] == selection_coeff]
    if not stored_data:  # stored_data is empty
        data = {'alpha': alpha,
                'selection_coefficient': selection_coeff, 'fixation': []}
        all_data.append(data)
        data_complete[1] = all_data

        with open(file_name, 'w') as write_file:
            json.dump(data_complete, write_file)


def generate_fixation(file_name,
                      pop_size,
                      alpha,
                      p0,
                      selection_coeff,
                      selection_type='fecundity',
                      model='cannings',
                      n_iter=1):
    """ Generate n_iter time to fixation or extinction and store them in file_name.

    >>> file_name = 'data_example/temporary.json'
    >>> exists(file_name) # there is no such file
    False
    >>> generate_fixation(file_name, pop_size=100, alpha=1.1, p0=0.1,
    ...                   selection_coeff=0.1, n_iter=10)
    >>> exists(file_name) # the file has been created and the data generated
    True
    >>> remove(file_name) # erasing the file
    """

    initialise_data(file_name, pop_size, alpha, p0,
                    selection_coeff, selection_type, model=model)

    with open(file_name, 'r') as read_file:
        data_complete = json.load(read_file)
        all_data = data_complete[1]

    stored_data = next(data for data in
                       all_data if (data['alpha'] == alpha
                                    and data['selection_coefficient'] == selection_coeff))
    id_stored_data = all_data.index(stored_data)

    sch = Schweinsberg(alpha, p0)
    for _ in range(n_iter):
        if selection_type == 'fecundity':
            fix, time = sch.fixation(pop_size, selection_fecundity=selection_coeff)
        elif selection_type == 'viability':
            fix, time = sch.fixation(pop_size, selection_viability=selection_coeff)
        else:
            raise Exception(
                f"The type of selection was {selection_type} but has to be "\
                 "'fecundity' or 'viability'")
        if not fix:
            time = -time
        stored_data['fixation'].append(time)

    all_data[id_stored_data] = stored_data
    data_complete[1] = all_data

    with open(file_name, 'w') as write_file:
        json.dump(data_complete, write_file)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
