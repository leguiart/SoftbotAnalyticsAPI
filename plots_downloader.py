
import base64
import time
import traceback
import json
import sys
import os
import logging
import argparse
import pandas as pd

from clients.softbot_analytics_client import SoftbotAnalyticsClient
from utils.concurrency_utils import generic_mm_concurrent_execution, generic_mm_parallel_execution

# create logger with __name__
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# create file handler which logs only warning level messages
fh = logging.FileHandler('downloader.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

STATISTICS = ["best",
              "worst",
              "std",
              "median",
              "average"]
ESTIMATORS = STATISTICS
MODES = ["bootstrap_dist",
         "est",
         "full"]
POPULATIONS = ["parent",
               "child",
               "default"]
N_BOOT = [20,50,100,2000,5000, "default"]
ARCHIVES = ["f_me_archive",
            "an_me_archive",
            "novelty_archive_un",
            "novelty_archive_an"]
ARCHIVE_INDICATORS = ["fitness",
            "unaligned_novelty",
            "aligned_novelty"]
EXPERIMENTS = ["ME","MNSLC","QN","NSLC","SO"]
# CHOOSE WINNER
WINNING_INDICATORS = [
    "fitness",
    "unaligned_novelty",
    "aligned_novelty",
    "gene_diversity",
    "control_gene_div",
    "morpho_gene_div",
    "morpho_div",
    "endpoint_div",
    "unaligned_novelty_archive_fit",
    "aligned_novelty_archive_fit",
    "unaligned_novelty_archive_novelty",
    "aligned_novelty_archive_novelty",
    "qd-score_ff",
    "qd-score_fun",
    "qd-score_fan",
    "qd-score_anf",
    "qd-score_anun",
    "qd-score_anan",
    "coverage"
    ]

NO_STD_INDICATORS = [
    "qd-score_ff",
    "qd-score_fun",
    "qd-score_fan",
    "qd-score_anf",
    "qd-score_anun",
    "qd-score_anan",
    "coverage"
    ]

# PAIRPLOTS
TASK_PERFORMANCE_INDICATORS = [
    "fitness",
    "aligned_novelty",
    "endpoint_div",
    "unaligned_novelty_archive_fit",
    "aligned_novelty_archive_fit",
    "qd-score_ff",
    "qd-score_fan",
    "qd-score_anf",
    "qd-score_anan",
]
PHENOTYPE_DIVERSITY_INDICATORS = [
    "morphology",
    "morphology_active",
    "morphology_passive",
    "morpho_div",
    "unaligned_novelty",
    "unaligned_novelty_archive_novelty",
    "aligned_novelty_archive_novelty",
    "qd-score_fun",
    "qd-score_anun",
    "coverage", 
]
GENE_DIVERSITY_INDICATORS = [
    "gene_diversity",
    "control_gene_div",
    "morpho_gene_div",
    "control_cppn_nodes",
    "control_cppn_edges",
    "control_cppn_ws",
    "morpho_cppn_nodes",
    "morpho_cppn_edges",
    "morpho_cppn_ws",
    "simplified_gene_div",
    "simplified_gene_ne_div",
    "simplified_gene_nws_div"
]
CORR_PLOT_INDICATORS = TASK_PERFORMANCE_INDICATORS + PHENOTYPE_DIVERSITY_INDICATORS + GENE_DIVERSITY_INDICATORS
# JOINTPLOTS
STATE_SPACE_INDICATORS = [
    "trayectory_div",
    "inipoint_x",
    "inipoint_y",
    "inipoint_z",
    "endpoint_x",
    "endpoint_y",
    "endpoint_z",
    "trayectory_x",
    "trayectory_y",
    "trayectory_z"
]
GRAPH_DESIGNSPACE_INDICATORS = [
    "control_cppn_nodes",
    "control_cppn_edges",
    "control_cppn_ws",
    "morpho_cppn_nodes",
    "morpho_cppn_edges",
    "morpho_cppn_ws"
]

# CONVERGENCE PLOTS, BOXPLOTS, VIOLINPLOTS, KDEs
ALL_INDICATORS = CORR_PLOT_INDICATORS + STATE_SPACE_INDICATORS.copy()
STATE_SPACE_INDICATORS = ["endpoint_div"] + STATE_SPACE_INDICATORS

def generate_parameters(param_key, modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    param_dict = {param_key : []}

    for mode in modes:
        args_list = [mode, indicators, statistics, experiments]
        for pop_type in populations:
            if pop_type != "default":
                kwargs_dict = {"population":pop_type}
            else:
                kwargs_dict = {}
            if mode == "bootstrap_dist":
                for n_boot in n_boots:
                    if n_boot != "default":
                        kwargs_dict["n_boot"] = n_boot
                    else:
                        del kwargs_dict["n_boot"]
                    kwargs_dict["lang"] = lang 
                    param_dict[param_key].append([args_list, kwargs_dict.copy()])
            elif mode == "est":
                for estimator in estimators:
                    if param_key == "kde_distributions" and estimator == "best" and pop_type == "parent":
                        continue
                    kwargs_dict["estimator"] = estimator
                    kwargs_dict["lang"] = lang 
                    param_dict[param_key].append([args_list, kwargs_dict.copy()])
            else:
                kwargs_dict["lang"] = lang 
                param_dict[param_key].append([args_list, kwargs_dict.copy()])
    
    return param_dict

def generate_kde_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters("kde_distributions",modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = lang)

def generate_boxplot_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters("boxplots",modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = lang)

def generate_violinplot_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters("violinplots", modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = lang)

def generate_convergence_params(indicators, statistics, experiments, populations, n_boots, lang = "es"):
    param_key = "convergence_plots" 
    param_dict = {param_key : []}

    args_list = [indicators, statistics, experiments]
    for pop_type in populations:
        if pop_type != "default":
            kwargs_dict = {"population":pop_type}
        else:
            kwargs_dict = {}
        for n_boot in n_boots:
            if n_boot != "default":
                kwargs_dict["n_boot"] = n_boot
            else:
                del kwargs_dict["n_boot"]
            kwargs_dict["lang"] = lang 
            param_dict[param_key].append([args_list, kwargs_dict.copy()])

    return param_dict

def generate_choosewinner_params(indicators, statistics, experiments, populations, lang = "es"):
    param_key = "choose_winner" 
    param_dict = {param_key : []}

    for pop_type in populations:
        if pop_type != "default":
            kwargs_dict = {"population":pop_type}
        else:
            kwargs_dict = {}
        kwargs_dict["lang"] = lang 
        for statistic in statistics:
            param_dict[param_key].append([[statistic, indicators, experiments], kwargs_dict])
    return param_dict

def kde_distributions(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    try:
        time.sleep(delay)
        httpRequester = SoftbotAnalyticsClient(base_url)
        img_dir_path = os.path.join(img_base_path, 'kde_distributions')
        logger.info(f'Start getting kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        response = httpRequester.kde_distributions(mode, indicators, statistics, experiment_names, **kwargs)
        logger.info(f'Finished getting kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        logged_response = {'msg' : response['msg']}
        logger.info(f'Result of kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
        imgs = response['img']
        if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
            os.mkdir(img_dir_path)
        logger.info(f'Saving figures: kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        img_name_prefix = "_".join(['KdePlot', mode,'-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()]])
        for i, indicator in enumerate(indicators):
            if len(imgs[i]) == 1:
                img_name = f'{img_name_prefix}_indicator={indicator}'
                img_data = bytes(imgs[i][0], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
                continue
            for j, statistic in enumerate(statistics):
                img_name = f'{img_name_prefix}_indicator={indicator}_statistic={statistic}'
                img_data = bytes(imgs[i][j], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
        logger.info(f'Finished saving figures: kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        return {
            'func' : 'kde_distributions',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs, 
            'response': logged_response
        }
    except Exception:
        traceback.print_exc()
        logger.exception(f'Failure retrieving kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
        return {
            'func' : 'kde_distributions',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs,
            'exception' : traceback.format_exc()
        }
    
def boxplots(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    try:
        time.sleep(delay)
        httpRequester = SoftbotAnalyticsClient(base_url)
        img_dir_path = os.path.join(img_base_path, 'boxplots')
        logger.info(f'Start getting box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        response = httpRequester.boxplots(mode, indicators, statistics, experiment_names, **kwargs)
        logger.info(f'Finished getting box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        logged_response = {'msg' : response['msg'], 'size' : response['msg'], 'format' : response['format']}
        logger.info(f'Result of box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
        imgs = response['img']
        if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
            os.mkdir(img_dir_path)
        logger.info(f'Saving figures: box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        img_name_prefix = "_".join(['BoxPlot', mode,'-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()]])
        for i, indicator in enumerate(indicators):
            if len(imgs[i]) == 1:
                img_name = f'{img_name_prefix}_indicator={indicator}'
                img_data = bytes(imgs[i][0], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
                continue
            for j, statistic in enumerate(statistics):
                img_name = f'{img_name_prefix}_indicator={indicator}_statistic={statistic}'
                img_data = bytes(imgs[i][j], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
        logger.info(f'Finished saving figures: box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        return {
            'func' : 'boxplots',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs, 
            'response': logged_response
        }
    except Exception:
        traceback.print_exc()
        logger.exception(f'Failure retrieving box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
        return {
            'func' : 'boxplots',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs,
            'exception' : traceback.format_exc()
        }

def violinplots(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    try:
        time.sleep(delay)
        httpRequester = SoftbotAnalyticsClient(base_url)
        img_dir_path = os.path.join(img_base_path, 'violinplots')
        logger.info(f'Start getting violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        response = httpRequester.violinplots(mode, indicators, statistics, experiment_names, **kwargs)
        logger.info(f'Finished getting violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        logged_response = {'msg' : response['msg'], 'size' : response['msg'], 'format' : response['format']}
        logger.info(f'Result of violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
        imgs = response['img']
        if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
            os.mkdir(img_dir_path)
        logger.info(f'Saving figures: violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        img_name_prefix = "_".join(['ViolinPlot', mode,'-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()]])
        for i, indicator in enumerate(indicators):
            if len(imgs[i]) == 1:
                img_name = f'{img_name_prefix}_indicator={indicator}'
                img_data = bytes(imgs[i][0], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
                continue
            for j, statistic in enumerate(statistics):
                img_name = f'{img_name_prefix}_indicator={indicator}_statistic={statistic}'
                img_data = bytes(imgs[i][j], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
        logger.info(f'Finished saving figures: violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
        return {
            'func' : 'violinplots',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs, 
            'response': logged_response
        }
    except Exception:
        traceback.print_exc()
        logger.exception(f'Failure retrieving violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
        return {
            'func' : 'violinplots',
            'args' : [base_url, mode, indicators, statistics, experiment_names],
            'kwargs' : kwargs,
            'exception' : traceback.format_exc()
        }
    
def convergence_plots(img_base_path, base_url, delay, indicators, statistics, experiment_names, n_boot='default', **kwargs):
    try:
        time.sleep(delay)
        httpRequester = SoftbotAnalyticsClient(base_url)
        img_dir_path = os.path.join(img_base_path, 'convergence_plots')
        logger.info(f'Start getting convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}')
        response = httpRequester.bs_convergence(indicators, statistics, experiment_names, n_boot=str(n_boot), **kwargs)
        logger.info(f'Finished getting convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}')
        logged_response = {'msg' : response['msg'], 'size' : response['size'], 'format' : response['format']}
        logger.info(f'Result of convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
        imgs = response['img']
        if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
            os.mkdir(img_dir_path)
        logger.info(f'Saving figures: convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}')
        img_name_prefix = "_".join(['ConvergencePlot', '-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()]])
        for i, indicator in enumerate(indicators):
            if len(imgs[i]) == 1:
                img_name = f'{img_name_prefix}_indicator={indicator}'
                img_data = bytes(imgs[i][0], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
                continue
            for j, statistic in enumerate(statistics):
                img_name = f'{img_name_prefix}_nboot={n_boot}_indicator={indicator}_statistic={statistic}'
                img_data = bytes(imgs[i][j], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
        logger.info(f'Finished saving figures: convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}')
        return {
            'func' : 'convergence_plots',
            'args' : [base_url,  indicators, statistics, experiment_names, n_boot],
            'kwargs' : kwargs, 
            'response': logged_response
        }
    except Exception:
        traceback.print_exc()
        logger.exception(f'Failure retrieving convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
        return {
            'func' : 'convergence_plots',
            'args' : [base_url,  indicators, statistics, experiment_names],
            'kwargs' : kwargs,
            'exception' : traceback.format_exc()
        }

def choose_winner(img_base_path, base_url, delay, statistic, indicators, experiment_names, **kwargs):
    try:
        time.sleep(delay)
        httpRequester = SoftbotAnalyticsClient(base_url)
        img_dir_path = os.path.join(img_base_path, 'choose_winners')
        logger.info(f'Start choose winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}')
        response = httpRequester.choose_winner(statistic, indicators, experiment_names, **kwargs)
        logger.info(f'Finished choose winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}')
        logged_response = {'msg' : response['msg'], 'condorcet_scores' : response['condorcet_scores'], 'borda_count' : response['borda_count']}
        logger.info(f'Result of choose winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
        if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
            os.mkdir(img_dir_path)
        logger.info(f'Saving choose winners tables with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}')
        tables_name = "_".join(['ChooseWinner', '-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()], statistic])
        condorcet_scores : dict[str, dict[str, int]]  = response['condorcet_scores']
        borda_count : dict[str, int] = response['borda_count']
        places_count  : dict[str, list[int]] = response['places_count']
        permutation_counts : dict[str, int] = response['permutation_counts']
        indicator_algo_scores : dict[str, dict[str, int]] = response['indicator_algo_scores']
        indicator_algo_tables = response['indicator_algo_tables']
        
        condorcet_scores_d : dict[str, list] = {"Algoritmo" : list(condorcet_scores[list(condorcet_scores.keys())[0]].keys()), **{k : list(map(lambda x : x[1], list(v.items()))) for k, v in condorcet_scores.items()}}
        condorcet_scores_table = pd.DataFrame(condorcet_scores_d)
        response['condorcet_scores_latex'] = condorcet_scores_table.to_latex(index=False, caption="Método de Condorcet")

        borda_count_d : dict[str, list] = {"Posición" : list(range(1, len(list(borda_count.keys())) + 1)) + ["Conteo de Borda"], **{k : places_count[k] + [borda_count[k]] for k in places_count.keys()}}
        borda_count_table = pd.DataFrame(borda_count_d)
        response["borda_count_latex"] = borda_count_table.to_latex(index=False, caption="Conteo de Borda")

        permutation_counts_table = pd.DataFrame([k.split(',') + [v] for k, v in permutation_counts.items()], columns=list(range(1, len(experiment_names) + 1)) + ["#Indicencias"])
        response["permutation_counts_latex"] = permutation_counts_table.to_latex(index=False, caption="Incidencia de permutaciones")

        indicator_algo_scores_d : dict[str, list]= dict(zip(["Indicador/Algoritmo"] + list(indicator_algo_scores[list(indicator_algo_scores.keys())[0]].keys()), 
                                                            [[indicator for indicator in indicator_algo_scores.keys()]] + 
                                                            [[score for score in indicator_algo_scores[indicator].values()]
                                                              for indicator in indicator_algo_scores.keys()]))
        indicator_algo_scores_table = pd.DataFrame(indicator_algo_scores_d)
        response["indicator_algo_scores_latex"] = indicator_algo_scores_table.to_latex(index=False, caption="Puntuaciones por algoritmo en cada indicador")

        indicator_algo_stats = dict(zip(["Indicador/Algoritmo"] + experiment_names, [[] for _ in range(len(experiment_names) + 1)]))
        for indicator, indicator_table in indicator_algo_tables.items():
            indicator_algo_stats["Indicador/Algoritmo"] += [indicator]
            for algo, val in indicator_table.items():
                mu = "{:.4f}".format(float(val[0][0]))
                sigma = "{:.4f}".format(float(val[0][1]))
                indicator_algo_stats[algo] += [f"{mu}({sigma})"]
        indicator_algo_stats_table = pd.DataFrame(indicator_algo_stats)
        response["indicator_algo_stats_latex"] = indicator_algo_stats_table.to_latex(index=False, caption="Indicadores por algoritmo")
        code_to_latex_symbols = {0 : "\\leftrightarrow", 1 : "\\downarrow", 2 : "\\uparrow"}
        inverted_code_to_latex_symbols = {0 : "\\leftrightarrow", 1 : "\\uparrow", 2 : "\\downarrow"}

        indicator_algo_tables_d = {}
        for indicator in indicator_algo_tables.keys(): 
            indicator_algo_d = {"Algoritmo" : experiment_names, **{exp : [] for exp in experiment_names}}
            for i in range(len(experiment_names)):
                algo1 = experiment_names[i]
                for j in range(len(experiment_names)):
                    algo2 = experiment_names[j]
                    if i == j:
                        cell_val = "--"
                    elif j < i:
                        indicator_algo_table = indicator_algo_tables[indicator][algo2][1]
                        val = indicator_algo_table[algo1]
                        p_val = val[2]
                        result = inverted_code_to_latex_symbols[val[3]]
                        cell_val = f"{p_val} {result}"
                    else:
                        indicator_algo_table = indicator_algo_tables[indicator][algo1][1]
                        val = indicator_algo_table[algo2]
                        p_val = val[2]
                        result = code_to_latex_symbols[val[3]]
                        cell_val = f"{p_val} {result}"
                    indicator_algo_d[algo1] += [cell_val]
            indicator_algo_table = pd.DataFrame(indicator_algo_d)
            indicator_algo_tables_d[indicator] = indicator_algo_table.to_latex(index=False, )
        response['indicator_algo_tables_latex'] = indicator_algo_tables_d
        json_object = json.dumps(response, indent=4)
        with open(f"{os.path.join(img_dir_path, tables_name)}.json", "w") as outfile:
            outfile.write(json_object)

        logger_cw = logging.getLogger(f"{__name__}.choose_winner")
        logger_cw.setLevel(logging.DEBUG)
        fh_cw = logging.FileHandler(f"{os.path.join(img_dir_path, tables_name)}.txt")
        fh_cw.setLevel(logging.INFO)
        logger_cw.addHandler(fh_cw)
        latex_tables_str = ""
        for l_key in response.keys():
            if l_key != 'indicator_algo_tables_latex':
                latex_tables_str += response[l_key]
            else:
                for indicator in response[l_key].keys():
                    latex_tables_str += response[l_key][indicator]
        logger_cw.info(latex_tables_str)

        logger.info(f'Finished saving choose winners tables with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}')
        return {
            'func' : 'choose_winners',
            'args' : [base_url,  indicators, statistic, experiment_names],
            'kwargs' : kwargs, 
            'response': logged_response
        }
    except Exception:
        traceback.print_exc()
        logger.exception(f'Failure choosing winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
        return {
            'func' : 'choose_winners',
            'args' : [base_url,  indicators, statistic, experiment_names],
            'kwargs' : kwargs,
            'exception' : traceback.format_exc()
        }

def json_to_func_params(img_base_path, base_url, delay, func_info : dict):
    func_params = []
    for func_name, func_params_list in func_info.items():
        func = globals()[func_name]
        func_params += [(func, [img_base_path, base_url, delay, *args], kwargs) for args, kwargs in func_params_list]
    return func_params

def main(parser : argparse.ArgumentParser):
    argv = parser.parse_args()
    host_url = argv.host
    func_info_path= argv.func_info_path
    p_num = int(argv.p_num)
    delay = float(argv.delay)
    _concurrent = argv.concurrent
    img_base_path = argv.img_base_path
    generate = argv.generate

    if generate:
        func_info = generate_kde_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)
        func_info = {**func_info, **generate_boxplot_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)}
        func_info = {**func_info, **generate_violinplot_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)}
        func_info = {**func_info, **generate_convergence_params(ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT)}
        func_info = {**func_info, **generate_choosewinner_params(WINNING_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS)}
        json_object = json.dumps(func_info, indent=4)
        with open(func_info_path, "w") as outfile:
            outfile.write(json_object)

    if os.path.exists(func_info_path):
        with open(func_info_path, "r") as outfile:
            func_info = json.load(outfile)
    else:
        logger.error(f'No file {func_info_path} exists')
        raise ValueError(f'No file {func_info_path} exists')

    func_params = json_to_func_params(img_base_path, host_url, delay, func_info)
    if _concurrent:
        retrieved_plots = generic_mm_concurrent_execution(func_params, p_num, 'Finished retrieving plot')
    else:
        retrieved_plots = generic_mm_parallel_execution(func_params, p_num, 'Finished retrieving plot')
    json_object = json.dumps(retrieved_plots, indent=4)
    with open("plots_downloader_finish.json", "w") as outfile:
        outfile.write(json_object)
    
if __name__ == '__main__':
    class CustomArgParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)
    arg_parser = CustomArgParser()
    arg_parser.add_argument('-f','--func_info_path', default='func.json', help = "JSON function info path")
    arg_parser.add_argument('-i','--img_base_path', default='.', help = "Path to save img info to")
    arg_parser.add_argument('-p','--p_num', default=5, help = "Number of processes/threads to attempt to launch")
    arg_parser.add_argument('--host', default='http://127.0.0.1:5000', help = "Host ip address or dns")
    arg_parser.add_argument('-d','--delay', default=0, help = "delay between process launch for parallel fp creation")
    arg_parser.add_argument('-c','--concurrent', action='store_true', help = "Use threads instead of isolated processes")
    arg_parser.add_argument('-g','--generate', action='store_true', help = "Generate func.json instead of using one provided from --func_info_path parameter")
    main(arg_parser)