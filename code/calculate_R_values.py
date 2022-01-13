import pandas as pd
import numpy as np
from os.path import join, isdir
from os import listdir
from multiprocess import Pool
import psutil
from tqdm import tqdm
import socket


def calculate_R_values(folder):
    files = listdir(folder)
    cutoffs = range(1, 101)

    cols = []
    for i in cutoffs:
        cols.extend([f"R_{i}_avg", f"R_{i}_std"])

    R_values = pd.DataFrame(columns=cols)
    R_values["R_1_avg"] = np.nan

    for j, file in enumerate(files):
        for cutoff in cutoffs:
            transmissions = pd.read_csv(join(src, folder, file))
            tmp = transmissions[transmissions["t"] <= cutoff]
            agg = (
                tmp[["ID", "target"]]
                .groupby("ID")
                .agg("count")
                .rename(columns={"target": "count"})
            )
            R_values.loc[j, f"R_{cutoff}_avg"] = agg["count"].mean()
            R_values.loc[j, f"R_{cutoff}_std"] = agg["count"].std()
    R_values.to_csv(join(src, folder + ".csv"), index=False)


hostname = socket.gethostname()
if hostname == "T14s":
    number_of_cores = 14  # laptop
    print("running on {}, using {} cores".format(hostname, number_of_cores))
elif hostname == "medea.isds.tugraz.at":
    number_of_cores = 200  # medea
    print("running on {}, using {} cores".format(hostname, number_of_cores))
else:
    number_of_cores = 1
    print("unknown host, using 1 core")

pool = Pool(number_of_cores)

src = "../data/simulation_results/omicron/ensembles_intervention_screening_omicron_all"
folders = [join(src, f) for f in listdir(src) if isdir(join(src, f))]

for folder in tqdm(
    pool.imap_unordered(func=calculate_R_values, iterable=folders), total=len(folders)
):
    pass
