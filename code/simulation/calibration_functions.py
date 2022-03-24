import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr

def weibull_two_param(shape, scale):
    '''
    A two-parameter Weibull distribution, based on numpy ramdon's single 
    parameter distribution. We use this distribution in the simulation to draw
    random epidemiological parameters for agents from the given distribution
    See https://numpy.org/doc/stable/reference/random/generated/numpy.random.weibull.html
    '''
    return scale * np.random.weibull(shape)


def get_weibull_shape(k, mu, var):
    '''
    Calculates the shape parameter of a Weibull distribution, given its mean
    mu and its variance var
    '''
    return var / mu**2 - gamma(1 + 2/k) / gamma(1+1/k)**2 + 1



def get_weibull_scale(mu, k):
    '''
    Calculates the scale parameter of a Weibull distribution, given its mean
    mu and its shape parameter k
    '''
    return mu / gamma(1 + 1/k)


def get_epi_params():
    '''
    Gets a combination of exposure duration, time until symptom onset and
    infection duration that satisfies all conditions.
    '''
    
    
    # shape and scale of Weibull distributions defined by the following means
    # and variances
    # exposure_duration = [5, 1.9] / days
    # time_until_symptoms = [6.4, 0.8] / days
    # infection_duration = [10.91, 3.95] / days
    epi_params = {
        'exposure_duration': [2.760216566831372, 8.033918996303989],
        'time_until_symptoms': [8.400996275174748, 6.727040406005965],
        'infection_duration': [1.8546626247614466, 9.458131298762975]}  

    tmp_epi_params = {}
    # iterate until a combination that fulfills all conditions is found
    while True:
        for param_name, param in epi_params.items():
            tmp_epi_params[param_name] = \
                round(weibull_two_param(param[0], param[1]))

        # conditions
        if tmp_epi_params['exposure_duration'] > 0 and \
           tmp_epi_params['time_until_symptoms'] >= \
           tmp_epi_params['exposure_duration'] and\
           tmp_epi_params['infection_duration'] > \
           tmp_epi_params['exposure_duration']:
           
            return tmp_epi_params
        

def get_outbreak_size_pdf(school_type, ensemble_results, outbreak_sizes):
    '''
    Extracts the discrite probability density function of outbreak sizes from
    the simulated and empirically measured outbreaks.
    
    Parameters:
    -----------
    school_type : string
        school type for which the distribution difference should be calculated.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc"
    ensemble_results : pandas DataFrame
        Data frame holding the results of the simulated outbreaks for a given
        school type and parameter combination. The outbreak size has to be given
        in the column "infected_total". 
    outbreak_size : pandas DataFrame
        Data frame holding the empirical outbreak size observations. The 
        outbreak size has to be given in the column "size", the school type in
        the column "type".
        
    Returns:
    --------
    simulation_pdf : numpy 1-d array
        Discrete probability density function of outbreak sizes from simulations
    empirical_pdf : numpy 1-d array
        Discrete probability density function of empirically observed outbreak
        sizes.
    '''
    # censor runs with no follow-up cases as we also do not observe these in the
    # empirical data
    ensemble_results = ensemble_results[ensemble_results['infected_total'] > 0].copy()

    obs = ensemble_results['infected_total'].value_counts()
    obs = obs / obs.sum()

    obs_dict = {size:ratio for size, ratio in zip(obs.index, obs.values)}

    # since we only have aggregated data for schools with and without daycare,
    # we map the daycare school types to their corresponding non-daycare types,
    # which are also the labels of the schools in the emirical data
    type_map = {'primary':'primary', 'primary_dc':'primary',
                'lower_secondary':'lower_secondary',
                'lower_secondary_dc':'lower_secondary',
                'upper_secondary':'upper_secondary',
                'secondary':'secondary', 'secondary_dc':'secondary'}
    school_type = type_map[school_type]

    expected_outbreaks = outbreak_sizes[\
                            outbreak_sizes['type'] == school_type].copy()
    expected_outbreaks.index = expected_outbreaks['size']

    exp_dict = {s:c for s, c in zip(range(1, expected_outbreaks.index.max() + 1), 
                                     expected_outbreaks['ratio'])}

    # add zeroes for both the expected and observed distributions in cases 
    # (sizes) that were not observed
    if len(obs) == 0:
        obs_max = 0
    else:
        obs_max = obs.index.max()

    for i in range(1, max(obs_max + 1,
                          expected_outbreaks.index.max() + 1)):
        if i not in obs.index:
            obs_dict[i] = 0
        if i not in expected_outbreaks.index:
            exp_dict[i] = 0

    simulation_pdf = np.asarray([obs_dict[i] for i in range(1, len(obs_dict) + 1)])
    empirical_pdf = np.asarray([exp_dict[i] for i in range(1, len(exp_dict) + 1)])
    
    return simulation_pdf, empirical_pdf


def get_group_case_pdf(school_type, ensemble_results, group_distributions):
    '''
    Extracts the ratios of simulated and empirically observed infected teachers
    and infected students for a given simulation parameter combination.
    
    Parameters
    ----------
    school_type : string
        school type for which the distribution difference should be calculated.
        Can be "primary", "primary_dc", "lower_secondary", "lower_secondary_dc",
        "upper_secondary", "secondary" or "secondary_dc"
    ensemble_results : pandas DataFrame
        Data frame holding the results of the simulated outbreaks for a given
        school type and parameter combination. The outbreak size has to be given
        in the column "infected_total". 
    group_distributions : pandas DataFrame
        Data frame holding the empirical observations of the ratio of infections
        in a given group (student, teacher) as compared to the overall number of
        infections (students + teachers). The data frame has three columns:
        "school_type", "group" and "ratio", where "group" indicates which group
        (student or teacher) the number in "ratio" belongs to. 
        
    Returns:
    --------
    simulation_pdf : numpy 1-d array
        Discrete probability density function of outbreak sizes from simulations
    empirical_pdf : numpy 1-d array
        Discrete probability density function of empirically observed outbreak
        sizes.
    '''
      
    # censor runs with no follow-up cases as we also do not observe these in the
    # empirical data
    ensemble_results = ensemble_results[ensemble_results['infected_total'] > 0].copy()
    
    # calculate ratios of infected teachers and students
    ensemble_results['teacher_ratio'] = ensemble_results['infected_teachers'] / \
                                        ensemble_results['infected_total'] 
    ensemble_results['student_ratio'] = ensemble_results['infected_students'] / \
                                        ensemble_results['infected_total'] 
    
    observed_distro = pd.DataFrame(\
        {'group':['student', 'teacher'],
         'ratio':[ensemble_results['student_ratio'].mean(),
                  ensemble_results['teacher_ratio'].mean()]})
    observed_distro = observed_distro.set_index('group')

    # since we only have aggregated data for schools with and without daycare,
    # we map the daycare school types to their corresponding non-daycare types,
    # which are also the labels of the schools in the emirical data
    type_map = {'primary':'primary', 'primary_dc':'primary',
                'lower_secondary':'lower_secondary',
                'lower_secondary_dc':'lower_secondary',
                'upper_secondary':'upper_secondary',
                'secondary':'secondary', 'secondary_dc':'secondary'}
    school_type = type_map[school_type]
    
    expected_distro = group_distributions[\
                            group_distributions['type'] == school_type].copy()
    expected_distro.index = expected_distro['group']
    
    simulation_pdf = np.asarray([observed_distro['ratio']['student'],
                                 observed_distro['ratio']['teacher']])
    empirical_pdf = np.asarray([expected_distro['ratio']['student'],
                                expected_distro['ratio']['teacher']])
    
    return simulation_pdf, empirical_pdf


def calculate_chi2_distance(simulation_pdf, empirical_pdf):
    '''
    Calculates the Chi-squared distance between the expected distribution of 
    outbreak sizes and the observed outbreak sizes in an ensemble of simulation 
    runs with the same parameters. 
    
    Parameters:
    -----------
    simulation_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in 
        the simulations. The index case needs to be subtracted from the pdf and 
        the pdf should be censored at 0 (as outbreaks of size 0 can not be 
        observed empirically).
    empirical_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in
        schools. Index cases are NOT included in outbreak sizes.
        
    Returns
    -------
    chi_2_distance : float
        Chi-squared distance between the simulated and empirically observed
        outbreak size distributions
    '''
    
    chi2_distance = ((empirical_pdf + 1) - (simulation_pdf + 1))**2 / \
            (empirical_pdf + 1)
    chi2_distance = chi2_distance.sum()
    
    return chi2_distance


def calculate_sum_of_squares_distance(simulation_pdf, empirical_pdf):
    '''
    Calculates the sum of squared distances between the expected distribution of 
    outbreak sizes and the observed outbreak sizes in an ensemble of simulation 
    runs with the same parameters. 
    
    Parameters:
    -----------
    simulation_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in 
        the simulations. The index case needs to be subtracted from the pdf and 
        the pdf should be censored at 0 (as outbreaks of size 0 can not be 
        observed empirically).
    empirical_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in
        schools. Index cases are NOT included in outbreak sizes.
        
    Returns:
    --------
    sum_of_squares : float
        sum of squared differences between the simulated and empirically 
        observed outbreak size distributions.
    '''
    sum_of_squares = ((empirical_pdf - simulation_pdf)**2).sum()    
    return sum_of_squares


def calculate_qq_regression_slope(simulation_pdf, empirical_pdf):
    '''
    Calculates the slope of a linear fit with intercept=0 to the qq plot of the 
    probability density function of the simulated values versus the pdf of the 
    empirically observed values. The number of quantiles is chosen to be 1/N,
    where N is the number of unique outbreak sizes observed in the simulation.
    Returns the absolute value of the difference between the slope of the fit 
    and a (perfect) slope of 1.
    
    Parameters:
    -----------
    simulation_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in 
        the simulations. The index case needs to be subtracted from the pdf and 
        the pdf should be censored at 0 (as outbreaks of size 0 can not be 
        observed empirically).
    empirical_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in
        schools. Index cases are NOT included in outbreak sizes.
        
    Returns:
    --------
    a : float
        Slope of the linear regression with intercept = 0 through the qq-plot
        of the simulated vs. the empirical discrete pdf.
    '''
    quant = 1 / len(simulation_pdf)
    simulation_quantiles = np.quantile(simulation_pdf, np.arange(0, 1, quant))
    empirical_quantiles = np.quantile(empirical_pdf, np.arange(0, 1, quant))
    a, _, _, _ = np.linalg.lstsq(simulation_quantiles[:, np.newaxis], empirical_quantiles,
                                 rcond=None)
    return np.abs(1 - a[0])


def calculate_pp_regression_slope(obs_cdf, exp_cdf):
    '''
    Calculates the slope of a linear fit with intercept=0 to the pp plot of the 
    cumulative probability density function of the simulated values versus the 
    cdf of the empirically observed values. Returns the absolute value of the
    difference between the slope of the fit and a (perfect) slope of 1.
    
    Parameters:
    -----------
    simulation_cdf : numpy 1-d array
        Discrete cumulative probability density function of the outbreak sizes 
        observed in the simulations. The index case needs to be subtracted from 
        the pdf before the cdf is calculated, and the pdf should be censored at 
        0 (as outbreaks of size 0 can not be observed empirically).
    empirical_pdf : numpy 1-d array
        Discrete cumulative probability density function of the outbreak sizes 
        observed in schools. Index cases are NOT included in the outbreak size 
        pdf from which the cdf was calculated.
        
    Returns:
    --------
    a : float
        Slope of the linear regression with intercept = 0 through the pp-plot
        of the simulated vs. the empirical discrete cdf.
    '''
    a, _, _, _ = np.linalg.lstsq(obs_cdf[:, np.newaxis], exp_cdf,
                                rcond=None)
    return np.abs(1 - a[0])


def calculate_bhattacharyya_distance(p, q):
    '''
    Calculates the Bhattacharyya distance between the discrete probability 
    density functions p and q.
    See also https://en.wikipedia.org/wiki/Bhattacharyya_distance).
    
    Parameters:
    -----------
    p, q : numpy 1-d array
        Discrete probability density function.
    empirical_pdf : numpy 1-d array
        Discrete probability density function of the outbreak sizes observed in
        schools. Index cases are NOT included in outbreak sizes.
        
    Returns:
    --------
    DB : float
        Bhattacharyya distance between the discrete probability 
        density functions p and q.
    '''
    BC = np.sqrt(p * q).sum()
    DB = - np.log(BC)
    return DB
    

def calculate_distances(ensemble_results, school_type, intermediate_contact_weight,
                       far_contact_weight, age_transmission_discount,
                       outbreak_size, group_distribution):
    
    # calculate the Chi-squared distance and the sum of squared differences
    # between the simulated and empirically observed ratios of teacher- and 
    # student cases
    simulation_group_distribution_pdf, empirical_group_distribution_pdf = \
        get_group_case_pdf(school_type, ensemble_results, group_distribution)
    chi2_distance_distro = calculate_chi2_distance(\
        simulation_group_distribution_pdf, empirical_group_distribution_pdf)
    sum_of_squares_distro = calculate_sum_of_squares_distance(\
        simulation_group_distribution_pdf, empirical_group_distribution_pdf)

    # calculate various distance measures between the simulated and empirically
    # observed outbreak size distributions
    simulation_outbreak_size_pdf, empirical_outbreak_size_pdf = \
        get_outbreak_size_pdf(school_type, ensemble_results, outbreak_size)
    simulation_outbreak_size_cdf = simulation_outbreak_size_pdf.cumsum()
    empirical_outbreak_size_cdf = empirical_outbreak_size_pdf.cumsum()
    
    # Chi-squared distance
    chi2_distance_size = calculate_chi2_distance(simulation_outbreak_size_pdf,
                                                 empirical_outbreak_size_pdf)
    # sum of squared differences
    sum_of_squares_size = calculate_sum_of_squares_distance(\
        simulation_outbreak_size_pdf, empirical_outbreak_size_pdf)
    # Bhattacharyya distance between the probability density functions
    bhattacharyya_distance_size = calculate_bhattacharyya_distance(\
        simulation_outbreak_size_pdf, empirical_outbreak_size_pdf)
    # Pearson correlation between the cumulative probability density functions
    pearsonr_size = np.abs(1 - pearsonr(simulation_outbreak_size_cdf, 
                             empirical_outbreak_size_cdf)[0])
    # Spearman correlation between the cumulative probability density functions
    spearmanr_size = np.abs(1 - spearmanr(simulation_outbreak_size_cdf, 
                             empirical_outbreak_size_cdf)[0])
    # Slope of the qq-plot with 0 intercept
    qq_slope_size = calculate_qq_regression_slope(simulation_outbreak_size_pdf,
                                                  empirical_outbreak_size_pdf)
    # Slope of the pp-plot with 0 intercept
    pp_slope_size = calculate_pp_regression_slope(simulation_outbreak_size_pdf,
                                                  empirical_outbreak_size_pdf)
    
    row = {
        'school_type':school_type,
        'intermediate_contact_weight':intermediate_contact_weight,
        'far_contact_weight':far_contact_weight,
        'age_transmission_discount':age_transmission_discount,
        'chi2_distance_distro':chi2_distance_distro,
        'sum_of_squares_distro':sum_of_squares_distro,
        'chi2_distance_size':chi2_distance_size,
        'sum_of_squares_size':sum_of_squares_size,
        'bhattacharyya_distance_size':bhattacharyya_distance_size,
        'pearsonr_difference_size':pearsonr_size,
        'spearmanr_difference_size':spearmanr_size,
        'qq_difference_size':qq_slope_size,
        'pp_difference_size':pp_slope_size,
        }
    return row

