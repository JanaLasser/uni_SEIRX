import pandas as pd
from os.path import join
import sys
import json

import data_creation_functions as dcf

N_runs = int(sys.argv[1])

testing = False
if N_runs == 1:
	testing = True

mode = 'no_intervention'
contact_network_src = '../data/networks'

params = [(N_runs, 
           False,
           False, 
           0.0,
           0.0,
           'overbooked',
           contact_network_type)
           for contact_network_type in ['all', 'TU', 'NaWi']]

for p in params:
    N_runs, u_mask, l_mask, u_vaccination_ratio,\
    l_vaccination_ratio, presence_fraction, contact_network_type = p
    
    dst = '../data/simulation_results/ensembles_{}_{}'\
        .format(mode, contact_network_type)
    
    dcf.run_ensemble(mode, N_runs, contact_network_src, 
            contact_network_type, dst, 
            u_mask=u_mask, l_mask=l_mask,
            u_vaccination_ratio=u_vaccination_ratio,
            l_vaccination_ratio=l_vaccination_ratio,
            presence_fraction=presence_fraction, testing=testing)

