import pandas as pd
from os.path import join
import sys
import json

import data_creation_functions as dcf

N_runs = int(sys.argv[1])

testing = False
if N_runs == 1:
	testing = True

screening_params = pd.read_csv(join('screening_params', 'measure_packages_delta.csv'))
mode = 'intervention_screening_delta'
#network_types = ['all', 'TU', 'NaWi']
network_types = ['all']

params = [(N_runs, 
           row['u_mask'],
           row['l_mask'], 
           row['u_vaccination_ratio'],
           row['l_vaccination_ratio'],
           row['presence_fraction'],
           contact_network_type)
           for i, row in screening_params.iterrows()\
           for contact_network_type in network_types]

print('there are {} different parameter combinations'.format(len(params)))


contact_network_src = '../../data/networks'

for p in params:
    N_runs, u_mask, l_mask, u_vaccination_ratio,\
    l_vaccination_ratio, presence_fraction, contact_network_type = p
    
    dst = '../../data/simulation_results/delta/ensembles_{}_{}'\
        .format(mode, contact_network_type)
    
    dcf.run_ensemble(mode, N_runs, contact_network_src, contact_network_type, dst, 
            u_mask=u_mask, l_mask=l_mask,
            u_vaccination_ratio=u_vaccination_ratio,
            l_vaccination_ratio=l_vaccination_ratio,
            presence_fraction=presence_fraction, testing=testing)
