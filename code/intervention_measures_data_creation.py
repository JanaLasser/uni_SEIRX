import pandas as pd
from os.path import join
import sys
import json

import data_creation_functions as dcf

with open('params/intervention_screening_measures.json', 'r') as fp:
    measures = json.load(fp)
with open('params/intervention_screening_simulation_parameters.json', 'r') as fp:
    simulation_params = json.load(fp)

N_runs = sys.argv[1]
screening_params = pd.read_csv(join('screening_params', 'measure_packages.csv'))

params = [(N_runs, 
           row['u_mask'],
           row['l_mask'], 
           row['presence_fraction'])
           for i, row in screening_params.iterrows()]

print('there are {} different parameter combinations'.format(len(params)))


contact_network_src = '../data/networks'
dst = '../data/simulation_results/ensembles'

results = pd.DataFrame()
for p in params:
    N_runs, unistudent_mask, lecturer_mask, presence_fraction = p
    
    dcf.run_ensemble(N_runs, measures,\
            simulation_params, contact_network_src, dst, 
            unistudent_mask=unistudent_mask, lecturer_mask=lecturer_mask,
            presence_fraction=presence_fraction, testing=False)