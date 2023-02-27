
import base64
import time
import traceback
import json
import sys
import os
import logging
import argparse
import pandas as pd
from collections import defaultdict

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

# Clasificacion acordada para elegir al ganador
# TASK_PERFORMANCE_INDICATORS = [
#     "fitness",
#     "aligned_novelty",
#     "unaligned_novelty_archive_fit",
#     "aligned_novelty_archive_fit",
#     "qd-score_ff",
#     "qd-score_fan",
#     "qd-score_anf",
#     "qd-score_anan",
# ]
# PHENOTYPE_DIVERSITY_INDICATORS = [
#     "morpho_div",
#     "unaligned_novelty",
#     "unaligned_novelty_archive_novelty",
#     "aligned_novelty_archive_novelty",
#     "qd-score_fun",
#     "qd-score_anun",
#     "coverage", 
# ]
# GENE_DIVERSITY_INDICATORS = [
#     "gene_diversity"
# ]

class KeyDict(defaultdict):
    def __missing__(self, key):
        return key

STATISTICS = ["best",
              "median",
              "average"]
ESTIMATORS = STATISTICS
MODES = ["bootstrap_dist",
         "est",
         "full"]
POPULATIONS = ["default"]
N_BOOT = [100, 1000, 2000]
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
MORPHO_DESIGNSPACE_INDICATORS = [
    "morphology",
    "morphology_active",
    "morphology_passive"
]
OBJECTIVESPACE_INDICATORS = [
    "fitness",
    "aligned_novelty",
    "unaligned_novelty"
]
UNINTERESTING_COMBINATIONS = [
    ["endpoint_div", "inipoint_x"],
    ["endpoint_div", "inipoint_y"],
    ["endpoint_div", "inipoint_z"],
    ["endpoint_div", "trayectory_x"],
    ["endpoint_div", "trayectory_y"],
    ["endpoint_div", "trayectory_z"],
    ["trayectory_div", "inipoint_x"],
    ["trayectory_div", "inipoint_y"],
    ["trayectory_div", "inipoint_z"],
    ["trayectory_div", "endpoint_x"],
    ["trayectory_div", "endpoint_y"],
    ["trayectory_div", "endpoint_z"],
    ["trayectory_div", "trayectory_x"],
    ["trayectory_div", "trayectory_y"],
    ["trayectory_div", "trayectory_z"],
    ["inipoint_x", "endpoint_x"],
    ["inipoint_x", "endpoint_y"],
    ["inipoint_x", "endpoint_z"],
    ["inipoint_x", "trayectory_x"],
    ["inipoint_x", "trayectory_y"],
    ["inipoint_x", "trayectory_z"],
    ["inipoint_y", "endpoint_y"],
    ["inipoint_y", "endpoint_y"],
    ["inipoint_y", "endpoint_z"],
    ["inipoint_y", "trayectory_y"],
    ["inipoint_y", "trayectory_y"],
    ["inipoint_y", "trayectory_z"],
    ["inipoint_z", "endpoint_z"],
    ["inipoint_z", "endpoint_z"],
    ["inipoint_z", "endpoint_z"],
    ["inipoint_z", "trayectory_z"],
    ["inipoint_z", "trayectory_z"],
    ["inipoint_z", "trayectory_z"],
    ["endpoint_x", "trayectory_y"],
    ["endpoint_x", "trayectory_z"],
    ["endpoint_y", "trayectory_x"],
    ["endpoint_y", "trayectory_z"],
    ["endpoint_z", "trayectory_x"],
    ["endpoint_z", "trayectory_y"],
    ["trayectory_x", "trayectory_y"],
    ["trayectory_x", "trayectory_z"],
    ["trayectory_y", "trayectory_z"],
]

# CONVERGENCE PLOTS, BOXPLOTS, VIOLINPLOTS, KDEs
ALL_INDICATORS = CORR_PLOT_INDICATORS + STATE_SPACE_INDICATORS.copy()
STATE_SPACE_INDICATORS = ["endpoint_div"] + STATE_SPACE_INDICATORS

def to_math_font(font, word, accent = '', suff = None):
    math_text = r"$" + accent + font + r"{" + word + r"}"
    if suff:
        math_text += r"^{suff}".format(suff = suff)
    math_text += r"$"
    return math_text

def to_math_it(word, accent = '', suff = None):
    return to_math_font(r"\mathit", word, accent=accent, suff=suff)

def to_math_cal(word, accent = '', suff = None):
    return to_math_font(r"\mathcal", word, accent=accent, suff=suff)

def to_math_bf(word, accent = '', suff = None):
    return to_math_font(r"\mathbf", word, accent=accent, suff=suff)

def to_math_frak(word, accent = '', suff = None):
    return to_math_font(r"\mathfrak", word, accent=accent, suff=suff)

def to_math_tt(word, accent = '', suff = None):
    return to_math_font(r"\mathtt", word, accent=accent, suff=suff)

def to_math_bb(word, accent = '', suff = None):
    return to_math_font(r"\mathbb", word, accent=accent, suff=suff)

def to_math_rm(word, accent = '', suff = None):
    return to_math_font(r"\mathrm", word, accent=accent, suff=suff)

def compactify_indicator(indicator):
    if any(prefix in indicator for prefix in ['parent', 'child']):
        subscript = indicator[0]
        suffix = "_".join(indicator.split('_')[1:])
    else:
        subscript = None
        suffix = indicator
    func = INDICATORS_TO_COMPACT[suffix][0]
    text = INDICATORS_TO_COMPACT[suffix][1]
    indicator_compact = func(text, suff=subscript)
    return indicator_compact


INDICATORS_TO_COMPACT = KeyDict(dict, 
            {"fitness" : (to_math_it, 'F'),
            "morphology" : (to_math_it, 'M'),
            "unaligned_novelty" : (to_math_it, 'N_u'),
            "aligned_novelty" : (to_math_it, 'N_a'), 
            "gene_diversity" : (to_math_it, 'D_g'),
            "control_gene_div" : (to_math_it, r'D_{gc}'),
            "morpho_gene_div" : (to_math_it, r'D_{gm}'),
            "morpho_div" : (to_math_it, 'D_m'),
            "endpoint_div" : (to_math_it, r'D_{\vec r(t=T)}'),
            "trayectory_div" : (to_math_it, r'D_{\vec s}'),
            "inipoint_x" : (to_math_it, r'\vec r_{x}(t=0)'),
            "inipoint_y": (to_math_it, r'\vec r_{y}(t=0)'),
            "inipoint_z": (to_math_it, r'\vec r_{z}(t=0)'),
            "endpoint_x": (to_math_it, r'\vec r_{x}(t=T)'),
            "endpoint_y": (to_math_it, r'\vec r_{y}(t=T)'),
            "endpoint_z": (to_math_it, r'\vec r_{z}(t=T)'),
            "trayectory_x": (to_math_it, r'(\vec s_{x}'),
            "trayectory_y": (to_math_it, r'\vec s_{y}'),
            "trayectory_z": (to_math_it, r'\vec s_{z}'),
            "morphology_active" : (to_math_it, 'M_a'),
            "morphology_passive" : (to_math_it, 'M_p'),
            "unaligned_novelty_archive_fit" : (to_math_tt, 'F_u'),
            "aligned_novelty_archive_fit" : (to_math_tt, 'F_a'),
            "unaligned_novelty_archive_novelty" : (to_math_tt, 'N_u'),
            "aligned_novelty_archive_novelty" : (to_math_tt, 'N_a'),
            "qd-score_ff" : (to_math_tt, r'Q_{ff}'),
            "qd-score_fun" : (to_math_tt, r'Q_{fun}'),
            "qd-score_fan" : (to_math_tt, r'Q_{fan}'),
            "qd-score_anf" : (to_math_tt, r'Q_{anf}'),
            "qd-score_anun" : (to_math_tt, r'Q_{anun}'),
            "qd-score_anan" : (to_math_tt, r'Q_{anan}'),
            "coverage" : (to_math_cal, "C"),            
            "control_cppn_nodes" : (to_math_rm, "C-CPPN_n"),
            "control_cppn_edges" : (to_math_rm, "C-CPPN_e"),
            "control_cppn_ws" : (to_math_rm, "C-CPPN_w"),
            "morpho_cppn_nodes" : (to_math_rm, "M-CPPN_n"),
            "morpho_cppn_edges" : (to_math_rm, "M-CPPN_e"),
            "morpho_cppn_ws" : (to_math_rm, "M-CPPN_w"),
            "simplified_gene_div" : (to_math_it, r"D_{gs}"),
            "simplified_gene_ne_div" : (to_math_it, r"D_{gcs}"),
            "simplified_gene_nws_div" : (to_math_it, r"D_{gms}")})

def generate_parameters(param_list : list, modes, indicators, statistics, experiments, populations, n_boots, estimators, lang, param_key):
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
                    param_list.append([args_list, kwargs_dict.copy()])
            elif mode == "est":
                for estimator in estimators:
                    if param_key == "kde_distributions" and estimator == "best" and pop_type == "parent":
                        continue
                    kwargs_dict["estimator"] = estimator
                    kwargs_dict["lang"] = lang 
                    param_list.append([args_list, kwargs_dict.copy()])
            else:
                kwargs_dict["lang"] = lang 
                param_list.append([args_list, kwargs_dict.copy()])

def generate_parameters2(param_list : list, modes, indicators, statistics, experiments, populations, n_boots, estimators, lang):
    for mode in modes:
        args_list = [mode, *indicators, statistics, experiments]
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
                    param_list.append([args_list, kwargs_dict.copy()])
            elif mode == "est":
                for estimator in estimators:
                    kwargs_dict["estimator"] = estimator
                    kwargs_dict["lang"] = lang 
                    param_list.append([args_list, kwargs_dict.copy()])
            else:
                kwargs_dict["lang"] = lang 
                param_list.append([args_list, kwargs_dict.copy()])

def generate_parameters_dict(param_key, modes, indicators, statistics, experiments, populations, n_boots, estimators, lang):
    param_dict = {param_key : []}
    generate_parameters(param_dict[param_key], modes, indicators, statistics, experiments, populations, n_boots, estimators, lang, param_key)
    return param_dict

def generate_kde_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters_dict("kde_distributions",modes, indicators, statistics, experiments, populations, n_boots, estimators, lang)

def generate_boxplot_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters_dict("boxplots",modes, indicators, statistics, experiments, populations, n_boots, estimators, lang)

def generate_violinplot_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    return generate_parameters_dict("violinplots", modes, indicators, statistics, experiments, populations, n_boots, estimators, lang)

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

def generate_pairplot_params(modes, indicator_groups, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    param_key = 'pair_plots'
    param_dict = generate_parameters_dict(param_key, modes, indicators, statistics, experiments, populations, n_boots, estimators, lang)
    for indicator_group in indicator_groups:
        generate_parameters(param_dict[param_key], modes, indicator_group, statistics, experiments, populations, n_boots, estimators, lang, param_key)
    for i, indicator_group1 in enumerate(indicator_groups):
        for j in range(i + 1, len(indicator_groups)):
            indicator_group2 = indicator_groups[j]
            indicator_pair = indicator_group1 + indicator_group2
            generate_parameters(param_dict[param_key], modes, indicator_pair, statistics, experiments, populations, n_boots, estimators, lang, param_key)
    return param_dict

def generate_jointplot_params(modes, indicators, statistics, experiments, populations, n_boots, estimators, lang = "es"):
    param_key = 'joint_plots'
    param_dict = {param_key : []}
    for i, indicator1 in enumerate(indicators):
        for j in range(i + 1, len(indicators)):
            indicator2 = indicators[j]
            if [indicator1, indicator2] not in UNINTERESTING_COMBINATIONS and [indicator2, indicator1] not in UNINTERESTING_COMBINATIONS:
                generate_parameters2(param_dict[param_key], modes, [indicator1, indicator2], statistics, experiments, populations, n_boots, estimators, lang)
    return param_dict

def generate_sarchive_params(archives, indicators, statistics, experiments, lang = "es"):
    param_key = "s_archive_plots"
    param_dict = {param_key : []}
    for archive in archives:
        for indicator in indicators:
            for statistic in statistics:
                args_list = [archive, indicator, statistic, experiments]
                kwargs_dict = {'lang' : lang}
                param_dict[param_key].append([args_list, kwargs_dict])
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
    retry_count = 0
    i = 0
    while(True):
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
            if i < retry_count:
                logger.warning(f'Failure retrieving kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving kde distributions with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'kde_distributions',
                'args' : [base_url, mode, indicators, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }
    
def boxplots(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
        try:
            time.sleep(delay)
            httpRequester = SoftbotAnalyticsClient(base_url)
            img_dir_path = os.path.join(img_base_path, 'boxplots')
            logger.info(f'Start getting box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            response = httpRequester.boxplots(mode, indicators, statistics, experiment_names, **kwargs)
            logger.info(f'Finished getting box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            logged_response = {'msg' : response['msg'], 'size' : response['size'], 'format' : response['format']}
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
            if i < retry_count:
                logger.warning(f'Failure retrieving boxplots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving box plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'boxplots',
                'args' : [base_url, mode, indicators, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }

def violinplots(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
        try:
            time.sleep(delay)
            httpRequester = SoftbotAnalyticsClient(base_url)
            img_dir_path = os.path.join(img_base_path, 'violinplots')
            logger.info(f'Start getting violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            response = httpRequester.violinplots(mode, indicators, statistics, experiment_names, **kwargs)
            logger.info(f'Finished getting violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            logged_response = {'msg' : response['msg'], 'size' : response['size'], 'format' : response['format']}
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
            if i < retry_count:
                logger.warning(f'Failure retrieving violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving violin plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'violinplots',
                'args' : [base_url, mode, indicators, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }
    
def convergence_plots(img_base_path, base_url, delay, indicators, statistics, experiment_names, n_boot='default', **kwargs):
    retry_count = 0
    i = 0
    while True:
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
            if i < retry_count:
                logger.warning(f'Failure retrieving convergence plots with:\nargs - {[base_url, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving convergence plots with:\nargs - {[base_url,  indicators, statistics, experiment_names, n_boot]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'convergence_plots',
                'args' : [base_url,  indicators, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }

def choose_winner(img_base_path, base_url, delay, statistic, indicators, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
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
            tables_name = "_".join(['ChooseWinner', '-'.join(experiment_names), statistic, *[f'{k}={v}' for k, v in kwargs.items()]])
            condorcet_scores : dict[str, dict[str, int]]  = response['condorcet_scores']
            borda_count : dict[str, int] = response['borda_count']
            places_count  : dict[str, list[int]] = response['places_count']
            permutation_counts : dict[str, int] = response['permutation_counts']
            indicator_algo_scores : dict[str, dict[str, int]] = response['indicator_algo_scores']
            indicator_algo_tables = response['indicator_algo_tables']

            condorcet_scores_d : dict[str, list] = {k : list(map(lambda x : x[1], list(v.items()))) for k, v in condorcet_scores.items()}
            condorcet_scores_table = pd.DataFrame(condorcet_scores_d, index = list(condorcet_scores.keys())).transpose()
            response['condorcet_scores_latex'] = condorcet_scores_table.to_latex(index=False, caption="Método de Condorcet",float_format="%.4f", escape=False)

            borda_count_d : dict[str, list] = {"Posición" : list(range(1, len(list(borda_count.keys())) + 1)) + ["Conteo de Borda"], **{k : places_count[k] + [borda_count[k]] for k in places_count.keys()}}
            borda_count_table = pd.DataFrame(borda_count_d)
            response["borda_count_latex"] = borda_count_table.to_latex(index=False, caption="Conteo de Borda",float_format="%.4f", escape=False)

            permutation_counts_table = pd.DataFrame([k.split(',') + [v] for k, v in permutation_counts.items()], columns=list(range(1, len(experiment_names) + 1)) + ["Indicencias"])
            response["permutation_counts_latex"] = permutation_counts_table.to_latex(index=False, caption="Incidencia de permutaciones",float_format="%.4f", escape=False)

            experiment_names_ias = list(indicator_algo_scores[list(indicator_algo_scores.keys())[0]].keys())
            indicator_algo_scores_d : dict[str, list]= dict(zip(["Indicador/Algoritmo"] + experiment_names_ias, 
                                                                [[compactify_indicator(indicator) for indicator in indicator_algo_scores.keys()]] + 
                                                                [[indicator_algo_scores[indicator][experiment] for indicator in indicator_algo_scores.keys()]
                                                                for experiment in experiment_names_ias]))
            indicator_algo_scores_table = pd.DataFrame(indicator_algo_scores_d)
            response["indicator_algo_scores_latex"] = indicator_algo_scores_table.to_latex(index=False, caption="Puntuaciones por algoritmo en cada indicador",float_format="%.4f", escape=False)
            
            # experiment_names_iats = list(list(indicator_algo_tables.items())[0][1].keys())
            indicator_algo_stats = dict(zip(["Indicador/Algoritmo"] + experiment_names, [[] for _ in range(len(experiment_names) + 1)]))
            for indicator, indicator_table in indicator_algo_tables.items():
                indicator_algo_stats["Indicador/Algoritmo"] += [compactify_indicator(indicator)]
                for algo, val in indicator_table.items():
                    mu = "{:.2f}".format(float(val[0][0]))
                    sigma = "{:.2f}".format(float(val[0][1]))
                    indicator_algo_stats[algo] += [f"{mu}({sigma})"]
            indicator_algo_stats_table = pd.DataFrame(indicator_algo_stats)
            response["indicator_algo_stats_latex"] = indicator_algo_stats_table.to_latex(index=False, caption="Indicadores por algoritmo",float_format="%.2f", escape=False)
            
            code_to_latex_symbols = {0 : r"$\leftrightarrow$", 1 : r"$\downarrow$", 2 : r"$\uparrow$"}
            inverted_code_to_latex_symbols = {0 : r"$\leftrightarrow$", 1 : r"$\uparrow$", 2 : r"$\downarrow$"}
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
                            cell_val = f"{'{:.4f}'.format(p_val)} {result}"
                        else:
                            indicator_algo_table = indicator_algo_tables[indicator][algo1][1]
                            val = indicator_algo_table[algo2]
                            p_val = val[2]
                            result = code_to_latex_symbols[val[3]]
                            cell_val = f"{'{:.4f}'.format(p_val)}  {result}"
                        indicator_algo_d[algo1] += [cell_val]
                indicator_algo_table = pd.DataFrame(indicator_algo_d)
                indicator_algo_tables_d[indicator] = indicator_algo_table.to_latex(index=False, float_format="%.4f", caption=f"Resultados de encuentros para {compactify_indicator(indicator)}", escape=False)
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
            for key in response.keys():
                if 'latex' in key and key != 'indicator_algo_tables_latex':
                    latex_tables_str += response[key]
                elif 'latex' in key:
                    for indicator in response[key].keys():
                        latex_tables_str += response[key][indicator]
            logger_cw.info(latex_tables_str)

            logger.info(f'Finished saving choose winners tables with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}')
            return {
                'func' : 'choose_winners',
                'args' : [base_url,  indicators, statistic, experiment_names],
                'kwargs' : kwargs, 
                'response': logged_response
            }
        except Exception:
            if i < retry_count:
                logger.warning(f'Failure choosing winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure choosing winners with:\nargs - {[base_url,  indicators, statistic, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'choose_winners',
                'args' : [base_url,  indicators, statistic, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }

def joint_plots(img_base_path, base_url, delay, mode, indicator1, indicator2, statistics, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
        try:
            time.sleep(delay)
            httpRequester = SoftbotAnalyticsClient(base_url)
            img_dir_path = os.path.join(img_base_path, 'jointkdeplots')
            logger.info(f'Start getting joint kde plots with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}')
            response = httpRequester.joint_kde(mode, indicator1, indicator2, statistics, experiment_names, **kwargs)
            logger.info(f'Finished getting joint kde plots with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}')
            logged_response = {'msg' : response['msg']}
            logger.info(f'Result of joint kde plot swith:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
            imgs = response['img']
            if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
                os.mkdir(img_dir_path)
            logger.info(f'Saving figures: joint kde plots with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}')
            img_name_prefix = "_".join(['JointPlot', mode,'-'.join(experiment_names), *[f'{k}={v}' for k, v in kwargs.items()]])
            for j, statistic in enumerate(statistics):
                img_name = f'{img_name_prefix}_indicator1={indicator1}_indicator2={indicator2}_statistic={statistic}'
                img_data = bytes(imgs[j], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')
            logger.info(f'Finished saving figure: joint kde plots with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}')
            return {
                'func' : 'joint_plots',
                'args' : [base_url, mode, indicator1, indicator2, statistics, experiment_names],
                'kwargs' : kwargs, 
                'response': logged_response
            }
        except Exception:
            if i < retry_count:
                logger.warning(f'Failure retrieving joint kde plot with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving joint kde plot with:\nargs - {[base_url, mode, indicator1, indicator2, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'joint_plots',
                'args' : [base_url, mode, indicator1, indicator2, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }
        
def pair_plots(img_base_path, base_url, delay, mode, indicators, statistics, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
        try:
            time.sleep(delay)
            httpRequester = SoftbotAnalyticsClient(base_url)
            img_dir_path = os.path.join(img_base_path, 'pair_plots')
            logger.info(f'Start getting pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            response = httpRequester.pairplot_kde(mode, indicators, statistics, experiment_names, **kwargs)
            logger.info(f'Finished getting pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            logged_response = {'msg' : response['msg'], 'corr_tables' : response['corr_tables']}
            logger.info(f'Result of pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
            pairplot_imgs = response['pairplot_imgs']
            corr_imgs = response['corr_imgs']
            corr_tables = response['corr_tables']
            if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
                os.mkdir(img_dir_path)
            logger.info(f'Saving figures: pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            img_name_prefix1 = "_".join(['PairPlot', mode,'-'.join(experiment_names), '-'.join(indicators), *[f'{k}={v}' for k, v in kwargs.items()]])
            img_name_prefix2 = "_".join(['Correlations', mode, '-'.join(indicators), *[f'{k}={v}' for k, v in kwargs.items()]])
            for j, statistic in enumerate(statistics):
                img_name1 = f'{img_name_prefix1}_statistic={statistic}'
                img_data1 = bytes(pairplot_imgs[j], 'utf-8')
                logger.info(f'Saving figure: {img_name1}')
                with open(os.path.join(img_dir_path, img_name1), "wb") as fh:
                    fh.write(base64.decodebytes(img_data1))
                logger.info(f'Saved figure: {img_name1}')
                for k, experiment in enumerate(experiment_names):
                    logger.info(f'Correlation table for pair plots with:\nargs - {[base_url, mode, indicators, statistic, experiment]}\nkwargs - {kwargs}\ncorr_table: {corr_tables[j][k]}')
                    img_name2 = f'{img_name_prefix2}_statistic={statistic}_experiment={experiment}'
                    img_data2 = bytes(corr_imgs[j][k], 'utf-8')
                    logger.info(f'Saving figure: {img_name2}')
                    with open(os.path.join(img_dir_path, img_name2), "wb") as fh:
                        fh.write(base64.decodebytes(img_data2))
                    logger.info(f'Saved figure: {img_name2}')
            logger.info(f'Finished saving figures: pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}')
            return {
                'func' : 'pair_plots',
                'args' : [base_url, mode, indicators, statistics, experiment_names],
                'kwargs' : kwargs, 
                'response': logged_response
            }
        except Exception:
            if i < retry_count:
                logger.warning(f'Failure retrieving pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving pair plots with:\nargs - {[base_url, mode, indicators, statistics, experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 'pair_plots',
                'args' : [base_url, mode, indicators, statistics, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }
    
def s_archive_plots(img_base_path, base_url, delay, archive, indicator, statistic, experiment_names, **kwargs):
    retry_count = 0
    i = 0
    while True:
        try:
            time.sleep(delay)
            httpRequester = SoftbotAnalyticsClient(base_url)
            img_dir_path = os.path.join(img_base_path, 's_archive_plots')
            logger.info(f'Start getting structured archive plots with:\nargs - {[base_url, archive, indicator, statistic, experiment_names]}\nkwargs - {kwargs}')
            response = httpRequester.structured_archive_plot(archive, indicator, statistic, experiment_names, **kwargs)
            logger.info(f'Finished getting structured archive plots with:\nargs - {[base_url, archive, indicator, statistic, experiment_names]}\nkwargs - {kwargs}')
            logged_response = {'msg' : response['msg'], 'size' : response['size'], 'format' : response['format']}
            logger.info(f'Result of structured archive plots with:\nargs - {[base_url, archive, indicator, statistic,experiment_names]}\nkwargs - {kwargs}\nresponse:\n{logged_response}')
            archive_imgs = response['img']
            if not (os.path.exists(img_dir_path) and os.path.isdir(img_dir_path)):
                os.mkdir(img_dir_path)
            logger.info(f'Saving figures: structured archive plots with:\nargs - {[base_url, archive, indicator, statistic, experiment_names]}\nkwargs - {kwargs}')
            img_name_prefix1 = "_".join(['SArchivePlot', archive, indicator, statistic, *[f'{k}={v}' for k, v in kwargs.items()]])
            for i, experiment in enumerate(experiment_names):
                img_name = f'{img_name_prefix1}_experiment={experiment}'
                img_data = bytes(archive_imgs[i], 'utf-8')
                logger.info(f'Saving figure: {img_name}')
                with open(os.path.join(img_dir_path, img_name), "wb") as fh:
                    fh.write(base64.decodebytes(img_data))
                logger.info(f'Saved figure: {img_name}')

            logger.info(f'Finished saving figures: structured archive plots with:\nargs - {[base_url,archive, indicator, statistic,experiment_names]}\nkwargs - {kwargs}')
            return {
                'func' : 's_archive_plots',
                'args' : [base_url,archive, indicator, statistic, experiment_names],
                'kwargs' : kwargs, 
                'response': logged_response
            }
        except Exception:
            if i < retry_count:
                logger.warning(f'Failure retrieving structured archive plots with:\nargs - {[base_url, archive, indicator, statistic,experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}\n Now waiting 30 [s] to try again')
                i+=1
                time.sleep(30)
                logger.info(f'Retry number: {i}')
                continue
            traceback.print_exc()
            logger.exception(f'Failure retrieving structured archive plots with:\nargs - {[base_url, archive, indicator, statistic,experiment_names]}\nkwargs - {kwargs}\nexception_info:\n {traceback.format_exc()}')
            return {
                'func' : 's_archive_plots',
                'args' : [base_url, archive, indicator, statistic, experiment_names],
                'kwargs' : kwargs,
                'exception' : traceback.format_exc()
            }

def json_to_func_params(img_base_path, base_url, delay, func_info : dict):
    func_params = []
    func_info_keys = list(func_info.keys())
    i = 0
    while func_info_keys:
        func_name = func_info_keys[i%len(func_info_keys)]
        func_params_list = func_info[func_name]
        # func_params += [(func, [img_base_path, base_url, delay, *args], kwargs) for args, kwargs in func_params_list]
        if len(func_params_list) > 0: 
            func = globals()[func_name]
            args, kwargs = func_params_list.pop(0)
            func_params += [(func, [img_base_path, base_url, delay*i, *args], kwargs)]
        else:
            func_info_keys.pop(i%len(func_info_keys))
        i += 1
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
        func_info = generate_convergence_params(ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT)
        # func_info = generate_choosewinner_params(WINNING_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS)
        # func_info = {**func_info, **generate_convergence_params(ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT)}
        # func_info = {**func_info, **generate_sarchive_params(ARCHIVES, ARCHIVE_INDICATORS, STATISTICS, EXPERIMENTS)}
        func_info = {**func_info, **generate_boxplot_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)}
        func_info = {**func_info, **generate_violinplot_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)}
        func_info = {**func_info, **generate_kde_params(MODES, ALL_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, N_BOOT, ESTIMATORS)}
        func_info = {**func_info, **generate_jointplot_params(MODES, STATE_SPACE_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, [N_BOOT[0]], [ESTIMATORS[2]])}
        func_info['joint_plots'] += generate_jointplot_params(MODES, GRAPH_DESIGNSPACE_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, [N_BOOT[0]], [ESTIMATORS[2]])['joint_plots']
        func_info['joint_plots'] += generate_jointplot_params(MODES, MORPHO_DESIGNSPACE_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, [N_BOOT[0]], [ESTIMATORS[2]])['joint_plots']
        func_info['joint_plots'] += generate_jointplot_params(MODES, OBJECTIVESPACE_INDICATORS, STATISTICS, EXPERIMENTS, POPULATIONS, [N_BOOT[0]], [ESTIMATORS[2]])['joint_plots']
        func_info = {**func_info, **generate_pairplot_params(MODES, [TASK_PERFORMANCE_INDICATORS, PHENOTYPE_DIVERSITY_INDICATORS, 
                                                                     GENE_DIVERSITY_INDICATORS], CORR_PLOT_INDICATORS, STATISTICS, 
                                                                     EXPERIMENTS, POPULATIONS, [N_BOOT[0]], ESTIMATORS)}
        for key in func_info.keys():
            print(f"({key}, {len(func_info[key])})")
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