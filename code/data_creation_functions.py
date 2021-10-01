import pandas as pd
from os.path import join
from os import listdir
import networkx as nx
import numpy as np

from scseirx.model_uni import SEIRX_uni
from scseirx import analysis_functions as af

# parallelisation functionality
from multiprocess import Pool
import psutil
from tqdm import tqdm

import socket

def compose_agents(measures, simulation_params):
    '''
    Utility function to compose agent dictionaries as expected by the simulation
    model as input from the dictionary of prevention measures.
    
    Parameters
    ----------
    prevention_measures : dictionary
        Dictionary of prevention measures. Needs to include the fields 
        (student, teacher, family_member) _screen_interval, index_probability
        and _mask. 
        
    Returns
    -------
    agent_types : dictionary of dictionaries
        Dictionary containing the fields "screening_interval", 
        "index_probability" and "mask" for the agent groups "student", "teacher"
        and "family_member".
    
    '''
    agent_types = {
            'unistudent':{
                'screening_interval':measures['unistudent_screen_interval'],
                'index_probability':simulation_params['unistudent_index_probability'],
                'mask':measures['unistudent_mask'],
                'vaccination_ratio':measures['unistudent_vaccination_ratio']},

            'lecturer':{
                'screening_interval': measures['lecturer_screen_interval'],
                'index_probability': simulation_params['lecturer_index_probability'],
                'mask':measures['lecturer_mask'],
                'vaccination_ratio':measures['lecturer_vaccination_ratio']},
    }
    
    return agent_types


def run_model(params):
    '''
    Runs a simulation with an SEIRX_school uni 
    
    Parameters:
    -----------
    params : tuple
        All model parameters
    Returns
    -------
    model : SEIRX_uni model instance holding a completed simulation run and
        all associated data.
    '''
    G, agent_types, measures, simulation_params, index_case, ttype,\
    u_screen_interval, l_screen_interval, unistudent_mask, lecturer_mask,\
    presence_fraction, ventilation_mod, seed = params

    N_steps=1000

    # initialize the model
    model = SEIRX_uni(G, 
      simulation_params['verbosity'], 
      base_transmission_risk = simulation_params['base_transmission_risk'],
      testing = measures['testing'],
      exposure_duration = simulation_params['exposure_duration'],
      time_until_symptoms = simulation_params['time_until_symptoms'],
      infection_duration = simulation_params['infection_duration'],
      quarantine_duration = measures['quarantine_duration'],
      subclinical_modifier = simulation_params['subclinical_modifier'],
      infection_risk_contact_type_weights = \
                 simulation_params['infection_risk_contact_type_weights'],
      K1_contact_types = measures['K1_contact_types'],
      diagnostic_test_type = measures['diagnostic_test_type'],
      preventive_screening_test_type = ttype,
      follow_up_testing_interval = \
                 measures['follow_up_testing_interval'],
      liberating_testing = measures['liberating_testing'],
      index_case = index_case,
      agent_types = agent_types, 
      age_transmission_risk_discount = \
                         simulation_params['age_transmission_discount'],
      age_symptom_modification = simulation_params['age_symptom_discount'],
      mask_filter_efficiency = simulation_params['mask_filter_efficiency'],
      transmission_risk_ventilation_modifier = ventilation_mod,
      seed=seed)

    # run the model until the outbreak is over
    for i in range(N_steps):
        # break if first outbreak is over
        if len([a for a in model.schedule.agents if \
            (a.exposed == True or a.infectious == True)]) == 0:
            break
        model.step()

    # collect the statistics of the single run
    row = af.get_ensemble_observables_uni(model, seed)
    row['seed'] = seed
    row['test_type'] = ttype
    row['unistudent_screen_interval'] = u_screen_interval
    row['lecturer_screen_interval'] = l_screen_interval
    row['unistudent_mask'] = unistudent_mask
    row['lecturer_mask'] = lecturer_mask
    row['presence_fraction'] = presence_fraction
    row['ventilation_mod'] = ventilation_mod
        
    return row


def run_ensemble(N_runs, measures, simulation_params, contact_network_src, 
              res_path, unistudent_mask=False, lecturer_mask=False, 
              presence_fraction=1, 
              ttype='same_day_antigen', u_screen_interval=None,
              l_screen_interval=None, ventilation_mod=1, testing=False):
    '''
    Utility function to run an ensemble of simulations for a given parameter 
    combination.
    
    Parameters:
    ----------
    N_runs : integer
        Number of individual simulation runs in the ensemble.
    measures : dictionary
        Dictionary listing all prevention measures in place for the given
        scenario. Fields that are not specifically included in this dictionary
        will revert to SEIRX_school defaults.
    simulation_params : dictionary
        Dictionary holding simulation parameters such as "verbosity" and
        "base_transmission_risk". Fields that are not included will revert back
        to SEIRX_school defaults.
    res_path : string
        Path to the directory in which results will be saved.
    contact_network_src : string
        Absolute or relative path pointing to the location of the contact
        network used for the calibration runs. The location needs to hold the
        contact networks for each school types in a sub-folder with the same
        name as the school type. Networks need to be saved in networkx's .bz2
        format.
    index_case : string
        Agent group from which the index case is drawn. Can be "unistudent" or
        "lecturer".
    ttype : string
        Test type used for preventive screening. For example "same_day_antigen"
    u_screen_interval : integer
        Interval between preventive screens in the unistudent agent group.
    l_screen_interval : integer
        Interval between preventive screens in the lecturer agent group.
    unistudent_mask : bool
        Wheter or not unistudents wear masks.
    lecturer_mask : bool
        Wheter or not lecturers wear masks.
    presence_fraction : float
        Fraction of students that are present in each lecture.
    ventilation_mod : float
        Modification to the transmission risk due to ventilation. 
        1 = no modification.
        
    Returns:
    --------
    ensemble_results : pandas DataFrame
        Data Frame holding the observable of interest of the ensemble, namely
        the number of infected unistudents and lecturers.
    '''

    # pick an index case with a probability for unistudents and lecturers
    # corresponding to an uniform distribution of infection probability
    # in the general population
    index_case = np.random.choice(['unistudent', 'lecturer'],
        p=[simulation_params['unistudent_index_probability'],
           simulation_params['lecturer_index_probability']])

    # create the agent dictionaries based on the given parameter values and
    # prevention measures
    agent_types = compose_agents(measures, simulation_params)
    agent_types['unistudent']['screening_interval'] = u_screen_interval
    agent_types['lecturer']['screening_interval'] = l_screen_interval
    agent_types['unistudent']['mask'] = unistudent_mask
    agent_types['lecturer']['mask'] = lecturer_mask

    # load the contact network, schedule and node_list corresponding to the school
    fname = 'university_2019-10-01_to_2019-10-07'
    if testing:
        fname = 'test'

    G = nx.readwrite.gpickle.read_gpickle(\
        join(contact_network_src, '{}_fraction-{}.bz2'\
            .format(fname, presence_fraction)))    

    turnovers = {'same':0, 'one':1, 'two':2, 'three':3}
    bmap = {True:'T', False:'F'}
    turnover, _, test = ttype.split('_')
    turnover = turnovers[turnover]
        
    measure_string = 'university_lmask-{}_umask-{}_pfrac-{}'\
        .format(bmap[lecturer_mask], bmap[unistudent_mask], presence_fraction)

    # figure out which host we are running on and determine number of cores to
    # use for the parallel programming
    hostname = socket.gethostname()
    if hostname == 'desiato':
        number_of_cores = 200 # desiato
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    elif hostname == 'T14s':
        number_of_cores = 14 # laptop
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    elif hostname == 'marvin':
        number_of_cores = 28 # marvin
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    elif hostname == 'medea.isds.tugraz.at':
        number_of_cores = 100 # medea
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    else:
        print('unknown host')

    pool = Pool(number_of_cores)

    params = [(G, agent_types, measures, simulation_params, index_case,
               ttype, u_screen_interval, l_screen_interval, unistudent_mask,
               lecturer_mask, presence_fraction, ventilation_mod, i) \
              for i in range(N_runs)]
    
    ensemble_results = pd.DataFrame()
    for row in tqdm(pool.imap_unordered(func=run_model,
                        iterable=params), total=len(params)):
        ensemble_results = ensemble_results.append(row, ignore_index=True)
        
    ensemble_results.to_csv(join(res_path, measure_string + '.csv'))
