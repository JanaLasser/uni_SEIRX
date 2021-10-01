import pandas as pd
from os.path import join
import sys
import json

import data_creation_functions as dcf

N_runs = int(sys.argv[1])

testing = False
if N_runs == 1:
	testing = True

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
    
    dcf.run_ensemble(N_runs, contact_network_src, dst, 
            unistudent_mask=unistudent_mask, lecturer_mask=lecturer_mask,
            presence_fraction=presence_fraction, testing=testing)
