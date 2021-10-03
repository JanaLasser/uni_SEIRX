import pandas as pd
from os.path import join
import sys
import json

import data_creation_functions as dcf

N_runs = int(sys.argv[1])

testing = False
if N_runs == 1:
	testing = True

mode = 'TTI'
contact_network_src = '../data/networks'
dst = '../data/simulation_results/ensembles_{}'.format(mode)
    
dcf.run_ensemble(mode, N_runs, contact_network_src, dst, testing)
