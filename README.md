# University COVID simulations
**Author:** Jana Lasser, TU Graz, Institute for interactive systems and data science (jana.lasser@tugraz.at)

**NOTE 1:** All data is available at https://doi.org/10.17605/OSF.IO/UPX7R. Simulation results are split into simulations with the Delta variant and simulations with the Omicron variant.  

**NOTE 2:** The code of the agent-based simulation framework is available in [this](https://github.com/JanaLasser/agent_based_COVID_SEIRX) repository. The present repository contains only the application to the university context.  

**NOTE 3:** The results pertaining to simulations with the Delta variant are described in [this](https://medrxiv.org/cgi/content/short/2021.11.16.21266383v1) preprint. The preprint for the results with the Omicron variant is upcoming.  

## Contact networks
Contact networks are created from raw enrolment data. At this point in time, I am not at liberty to share this data.  

Data cleaning is performed in the script `clean_data.ipynb`. The script processes a range of tables that were exported from the TU Graz lecture management system:  
* Information about which students are enrolled in which lectures and tutorials
* Information about which students are enrolled in which exams
* Information about which lecturers are responsible for which lectures and exams
* Information about which student is enrolled in which study
* Information about which lecturer is affiliated with which institute of the university
* Information about dates and times of lectures, tutorials and exams
* Information about the rooms the lectures, tutorials and exams take place in

The cleaned data is processed by the script `create_network.ipynb` to calculate some basic statistics about the data and create the co-location networks of students and lecturers. The main functionality for the network creation is offloaded into the library `network_creation_functions.py`. The script creates a variety of different networks:
* Networks with 25% occupancy, 50% occupancy, 100% occupancy and "overbooked" courses
* Networks with all students, unly TU Graz students and only NaWi students (see supporting information of the publication for further details)

The difference between 100% occupancy and "overbooked" is that in the "overbooked" networks, all students that enrolled in a lecture are included in the contact network, even if their number surpasses the number of seats available in a given lecture hall. In the 100% occupancy networks, students are removed from the network at random, until the seating capacity is reached.  

Different network observables (for example the node degrees) are analyzed in the script `analysis_network.ipynb`.  

Network visualizations (hairballs) are created in the script `visualize_network.ipynb`.


## Calibration
Since there is no direct observational data of outbreaks in the university context available to us, we have to resort to the model calibration in the school context. The calibration procedure is described in great detail in the school [repository](https://github.com/JanaLasser/agent_based_COVID_SEIRX#calibration-for-schools) and [paper](https://doi.org/10.1101/2021.04.13.21255320).

The calibration is adapted here such that it is only performed for secondary and upper secondary school types, since these contexts are closest to the university setting. The calibration is performed in the script `calibration.ipynb` and the main calibration functionality is offloaded into the library `calibraiton_functions.py`.

## Agent-based simulations
The calibrated model together with the contact networks are used to study the transmission dynamics at TU Graz given different intervention scenarios. Scpecifically, we study
* different lecture hall occupancies of 25%, 50% and "overbooked" (called 100% in the preprint)
* whether or not everybody is wearing a mask
* different levels of vaccine effectiveness against infection with the virus: 0%, 30%, 50% and 70% (we actually study all effectiveness levels between 0% and 100% in 10% steps but we don't report all of those).

Before we study any interventions, we perform simulations of two types of baseline scenarios:
* No intervention measures (`no_measure_delta_data_creation.py` and `no_measure_omicron_data_creation.py`) and
* Only test-trace-isolate (`TTI_delta_data_creation.py` and `TTI_omicron_data_creation.py`).

We split the scripts between simulations for the omicron and delta variants. Each of the scripts reads the parameters for the simulations and the basic intervention measures for the given scenario from `.json` files in the `/params` folder. Each simulation requires both a `_measures.json` and a `_simulation_parameters.json` file. In addition, the scripts also read a list of parameters that are varied from the folder `screening_params`. For the "no intervention" and "TTI" scenarios, only the effectiveness of the vaccine against infection is varied.  

The scripts are meant to run on a server with a large number of cores. While they could be run on a consumer grade machine in principle, they will take a very long time to complete. 

Simulations for the different intervention measure scenarios are then performed in the scripts `intervention_measure_delta_data_creation.py` and `intervention_measure_omicron_data_creation.py`.  Similar to the "no intervention" and "TTI" scenarios, the simulation scripts read their parameters and basic measures from `.json` files, and get a list of different parameter combinations for different intervention scenarios from `.csv` files in the `screening_params` folder. 

A lot of shared functionality between the different data screation scripts is offloaded to the `data_creation_functions.py` library.  

Each data creation script takes a single integer number as command line input, which specifies the number of simulations that are run for each distinct parameter combination. For the publication, we create ensembles of 1000 simulations. This takes about three days for all parameter combinations, running on 200 threads cores of [AMD epyc](https://www.amd.com/en/processors/epyc-7002-series) processors. 

Each data creation script creates a subfolder in the `/data/simulation_results/[VARIANT]/` directory and saves ensemble level statistics there. We make these simulation results available in the [OSF data repository](https://doi.org/10.17605/OSF.IO/UPX7R).  

In addition to ensemble-level statistics, the different data creation scripts also create files with the full transmission chains for each simulation. These files are not shared in the [OSF data repository](https://doi.org/10.17605/OSF.IO/UPX7R), because it would be too much data. These files are processed by the script `calculate_R_values.py`, to calculate the values of $R_\mathrm{eff}$, which are in turn saved as ensemble level statistics (subscript `_transmissions`). These files are uploaded in the [OSF data repository](https://doi.org/10.17605/OSF.IO/UPX7R). 

## Data analysis
The ensemble statistics created by the simulations are processed by a number of analysis scripts that follow a similar structure as the different simulation scenarios:
* `analysis_no_measures.ipynb` provides the analysis of the "no intervention" scenarios (for Delta and Omicron)
* `analysis_TTI.ipynb` provides the analysis for the "TTI" scenarios (for Delta and Omicron)
* `analysis_intervention_measures_delta.ipynb` and `analysis_intervention_measures_omicron.ipynb` provide the analysis for the different intervention measure scenarios (this time split for Delta and Omicron).

All scripts create plots that are saved in the folder `/plots`. 