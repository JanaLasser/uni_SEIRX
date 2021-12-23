import pandas as pd
from os.path import join, exists
from os import listdir, mkdir
import networkx as nx
import numpy as np
import json

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
        (student, teacher, family_member) _screen_interval
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
                'mask':measures['unistudent_mask'],
                'vaccination_ratio':measures['unistudent_vaccination_ratio']},

            'lecturer':{
                'screening_interval': measures['lecturer_screen_interval'],
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

    mode, contact_network_src, contact_network_type, testing, \
    u_vaccination_ratio, l_vaccination_ratio, \
    u_screen_interval, l_screen_interval, u_mask, l_mask, \
    presence_fraction, vaccination_modification, seed = params

    with open('params/{}_measures.json'.format(mode), 'r') as fp:
        measures = json.load(fp)
    with open('params/{}_simulation_parameters.json'.format(mode), 'r') as fp:
        simulation_params = json.load(fp)

    measures['unistudent_vaccination_ratio'] = u_vaccination_ratio
    measures['lecturer_vaccination_ratio'] = l_vaccination_ratio
    simulation_params["transmission_risk_vaccination_modifier"] = \
         {'reception':vaccination_modification, 'transmission':0}

    # create the agent dictionaries based on the given parameter values and
    # prevention measures
    agent_types = compose_agents(measures, simulation_params)
    agent_types['unistudent']['screening_interval'] = u_screen_interval
    agent_types['lecturer']['screening_interval'] = l_screen_interval
    agent_types['unistudent']['mask'] = u_mask
    agent_types['lecturer']['mask'] = l_mask

    # load the contact network, schedule and node_list corresponding to the school
    fname = 'university_2019-10-16_to_2019-10-23'
    if testing:
        fname = 'test'

    G = nx.readwrite.gpickle.read_gpickle(\
        join(contact_network_src, '{}_fraction-{}_{}.bz2'\
            .format(fname, presence_fraction, contact_network_type))) 

    # pick an index case with a probability for unistudents and lecturers
    # corresponding to an uniform distribution of infection probability
    # in the general population
    N_students = len([n for n in G.nodes(data='type') if n[1] == 'unistudent'])
    N_lecturers = len([n for n in G.nodes(data='type') if n[1] == 'lecturer'])
    p_student = N_students / (N_students + N_lecturers)
    p_lecturer = N_lecturers / (N_students + N_lecturers)

    index_case = np.random.choice(['unistudent', 'lecturer'], 
        p=[p_student, p_lecturer]) 

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
      preventive_screening_test_type = \
                 measures['preventive_screening_test_type'],
      follow_up_testing_interval = \
                 measures['follow_up_testing_interval'],
      liberating_testing = measures['liberating_testing'],
      index_case = index_case,
      agent_types = agent_types, 
      age_transmission_risk_discount = \
                         simulation_params['age_transmission_discount'],
      age_symptom_modification = simulation_params['age_symptom_discount'],
      mask_filter_efficiency = simulation_params['mask_filter_efficiency'],
      transmission_risk_ventilation_modifier = \
                 measures['transmission_risk_ventilation_modifier'],
      transmission_risk_vaccination_modifier = \
                 simulation_params['transmission_risk_vaccination_modifier'],
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
    row['index_case'] = index_case
    row['unistudent_screen_interval'] = u_screen_interval
    row['lecturer_screen_interval'] = l_screen_interval
    row['unistudent_mask'] = u_mask
    row['lecturer_mask'] = l_mask
    row['unistudent_vaccination_ratio'] = u_vaccination_ratio
    row['lecturer_vaccination_ratio'] = l_vaccination_ratio
    row['presence_fraction'] = presence_fraction
    row['testing'] = measures['testing']
    row['transmission_risk_ventilation_modifier'] =\
        measures['transmission_risk_ventilation_modifier'],
    row['transmission_risk_vaccination_modifier'] =\
        simulation_params['transmission_risk_vaccination_modifier']
    
    
    # unvaccinated lecturers
    row['unvaccinated_lecturers'] = \
        len([a for a in model.schedule.agents if a.type == "lecturer" and \
             a.vaccinated==False])
    # vaccinated lecturers
    row['vaccinated_lecturers'] = \
        len([a for a in model.schedule.agents if a.type == "lecturer" and \
             a.vaccinated==True])
    # infected unvaccinated lecturers
    row['infected_unvaccinated_lecturers'] = \
        len([a for a in model.schedule.agents if a.type == "lecturer" and \
             a.vaccinated==False and a.recovered==True])
    # infected vaccinated lecturers
    row['infected_vaccinated_lecturers'] = \
        len([a for a in model.schedule.agents if a.type == "lecturer" and \
             a.vaccinated==True and a.recovered==True])
    
    # unvaccinated unistudents
    row['unvaccinated_unistudents'] = \
        len([a for a in model.schedule.agents if a.type == "unistudents" and \
             a.vaccinated==False])
    # vaccinated unistudents
    row['vaccinated_unistudents'] = \
        len([a for a in model.schedule.agents if a.type == "unistudents" and \
             a.vaccinated==True])
    # infected unvaccinated unistudents
    row['infected_unvaccinated_unistudents'] = \
        len([a for a in model.schedule.agents if a.type == "unistudents" and \
             a.vaccinated==False and a.recovered==True])
    # infected vaccinated unistudents
    row['infected_vaccinated_unistudents'] = \
        len([a for a in model.schedule.agents if a.type == "unistudents" and \
             a.vaccinated==True and a.recovered==True])
        
    return row


def run_ensemble(mode, N_runs, contact_network_src, contact_network_type, res_path, 
                 u_mask=False, l_mask=False, u_vaccination_ratio=0.0,
                 l_vaccination_ratio=0.0, presence_fraction=1.0, 
                 u_screen_interval=None, l_screen_interval=None, testing=False,
                 vaccination_modification=0.47):
    '''
    Utility function to run an ensemble of simulations for a given parameter 
    combination.
    
    Parameters:
    ----------
    N_runs : integer
        Number of individual simulation runs in the ensemble.
    contact_network_src : string
        Absolute or relative path pointing to the location of the contact
        network used for the calibration runs. The location needs to hold the
        contact networks for each school types in a sub-folder with the same
        name as the school type. Networks need to be saved in networkx's .bz2
        format.
    contact_network_type : string
        Type of the contact network. Can be "all" (all students), "TU" (only
        TU Graz students) and "NaWi" (only NaWi students, i.e. students that
        are shared with KFU Graz).
    res_path : string
        Path to the directory in which results will be saved.
    u_mask : bool
        Wheter or not unistudents wear masks.
    l_mask : bool
        Wheter or not lecturers wear masks.
    u_vaccination_ratio : float
        Ratio of vaccinated unistudents.
    l_vaccination_ratio : float
        Ratio of vaccinated lecturers.
    presence_fraction : float
        Fraction of students that are present in each lecture.
    u_screen_interval : integer
        Interval between preventive screens in the unistudent agent group.
    l_screen_interval : integer
        Interval between preventive screens in the lecturer agent group.
    testing : bool
        Whether or not the simulation is running tests to detect infections.
    vaccination_modification : float
        Effectiveness of vaccinations against infection.
        
    Returns:
    --------
    ensemble_results : pandas DataFrame
        Data Frame holding the observable of interest of the ensemble, namely
        the number of infected unistudents and lecturers.
    '''  

    bmap = {True:'T', False:'F'}
    measure_string = 'university_lmask-{}_umask-{}_pfrac-{}'\
        .format(bmap[l_mask], bmap[u_mask], presence_fraction) +\
        '_uvacc-{}_lvacc-{}'.format(u_vaccination_ratio, l_vaccination_ratio) +\
        '_vaccmod-{}'.format(vaccination_modification)

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
        number_of_cores = 200 # medea
        print('running on {}, using {} cores'.format(hostname, number_of_cores))
    else:
        print('unknown host')

    pool = Pool(number_of_cores)

    params = [(mode, contact_network_src, contact_network_type, testing,
               u_vaccination_ratio, l_vaccination_ratio,
               u_screen_interval, l_screen_interval, 
               u_mask, l_mask, presence_fraction, vaccination_modification, i) \
            for i in range(N_runs)]
    
    ensemble_results = pd.DataFrame()
    for row in tqdm(pool.imap_unordered(func=run_model,
                        iterable=params), total=len(params)):
        ensemble_results = ensemble_results.append(row, ignore_index=True)
        
    if not exists(res_path):
        mkdir(res_path)

    ensemble_results.to_csv(join(res_path, measure_string + '.csv'),
        index=False)

    