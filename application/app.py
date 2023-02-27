
from ast import Dict
import itertools
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp

from scipy import stats
from PIL import Image
from flask_cors import CORS
from flask import Flask, jsonify, request
from collections import defaultdict

# from application.data.dal import Dal # debug
from data.dal import Dal

class KeyDict(defaultdict):
    def __missing__(self, key):
        return key

INDICATOR_STATS_SET = [
            "fitness",
            "morphology",
            "unaligned_novelty",
            "aligned_novelty",
            "gene_diversity",
            "control_gene_div",
            "morpho_gene_div",
            "morpho_div",
            "endpoint_div",
            "trayectory_div",
            "inipoint_x",
            "inipoint_y",
            "inipoint_z",
            "endpoint_x",
            "endpoint_y",
            "endpoint_z",
            "trayectory_x",
            "trayectory_y",
            "trayectory_z",
            "morphology_active",
            "morphology_passive",
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
            "coverage",            
            "control_cppn_nodes",
            "control_cppn_edges",
            "control_cppn_ws",
            "morpho_cppn_nodes",
            "morpho_cppn_edges",
            "morpho_cppn_ws",
            "simplified_gene_div",
            "simplified_gene_ne_div",
            "simplified_gene_nws_div"]

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

PLOT_INDICATORS = [
            "fitness",
            "morphology",
            "unaligned_novelty",
            "aligned_novelty",
            "gene_diversity",
            "control_gene_div",
            "morpho_gene_div",
            "morpho_div",
            "endpoint_div",
            "morphology_active",
            "morphology_passive",
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
            "coverage",            
            "control_cppn_nodes",
            "control_cppn_edges",
            "control_cppn_ws",
            "morpho_cppn_nodes",
            "morpho_cppn_edges",
            "morpho_cppn_ws",
            "simplified_gene_div",
            "simplified_gene_ne_div",
            "simplified_gene_nws_div"]

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
            "coverage"]

NO_STD_INDICATORS = [
    "qd-score_ff",
    "qd-score_fun",
    "qd-score_fan",
    "qd-score_anf",
    "qd-score_anun",
    "qd-score_anan",
    "coverage"]

ARCHIVES = ["f_me_archive",
            "an_me_archive",
            "novelty_archive_un",
            "novelty_archive_an"]

STATISTICS = ["best",
              "worst",
              "average",
              "std",
              "median"]

LANG_DICTS = {
    "en" : KeyDict(dict, {
        "archives_title_template" : "{indicator} {statistic} feature map for {experiment_name} experiment",
        "jointkde_title_template_est" : "Joint KDE of {estimator} of generational {statistic}",
        "jointkde_title_template_boot" : "Joint KDE of bootstrapped (n={n_boot}) generational {statistic}",
        "jointkde_title_template_all" : "Joint KDE plots of full generational {statistic} distribution",
        "pairplot_title_template_est" : "Regression/Scatter plots of {estimator} of generational {statistic}",
        "pairplot_title_template_boot" : "Regression/Scatter plots of bootstrapped (n={n_boot}) generational {statistic}",
        "pairplot_title_template_all" : "Regression/Scatter plots of full generational {statistic} distribution",
        "corr_title_template_est" : "Correlations of {estimator} of generational {statistic} ({experiment_name})",
        "corr_title_template_boot" : "Correlations of bootstrapped (n={n_boot}) generational {statistic} ({experiment_name})",
        "corr_title_template_all" : "Correlationns of full generational {statistic} distribution ({experiment_name})",
        "box_title_template_est" : "Boxplot of {estimator} of generational {indicator} {statistic}",
        "box_title_template_boot" : "Boxplot of bootstrapped (n={n_boot}) generational {indicator} {statistic}",
        "box_title_template_all" : "Boxplot of full generational {indicator} {statistic} distribution",
        "violin_title_template_est" : "Violinplot of {estimator} of generational {indicator} {statistic}",
        "violin_title_template_boot" : "Violinplot of bootstrapped (n={n_boot}) generational {indicator} {statistic}",
        "violin_title_template_all" : "Violinplot of full generational {indicator} {statistic} distribution",
        "kde_title_template_est" : "KDE plot of {estimator} of generational {indicator} {statistic}",
        "kde_title_template_boot" : "KDE plot of bootstrapped (n={n_boot}) generational {indicator} {statistic}",
        "kde_title_template_all" : "KDE plot of full generational {indicator} {statistic} distribution",

    }),
    "es" : KeyDict(dict, {
        "archives_title_template" : "Mapa de características para {statistic} de {indicator} del experimento {experiment_name}",
        "jointkde_title_template_est" : "KDEs conjuntas de {estimator} de {statistic} generacional",
        "jointkde_title_template_boot" : "KDEs conjuntas de {statistic} generacional con bootstrap (n={n_boot}) ",
        "jointkde_title_template_all" : "KDEs conjuntas de la distribución completa de {statistic} generacional",
        "pairplot_title_template_est" : "Regresión/dispersión de {estimator} de {statistic} generacional",
        "pairplot_title_template_boot" : "Regresión/dispersión de {statistic} generacional con bootstrap (n={n_boot}) ",
        "pairplot_title_template_all" : "Regresión/dispersión de la distribución completa de {statistic} generacional",
        "corr_title_template_est" : "Correlaciones de {estimator} de {statistic} generacional ({experiment_name})",
        "corr_title_template_boot" : "Correlaciones de {statistic} generacional con bootstrap (n={n_boot}) ({experiment_name})",
        "corr_title_template_all" : "Correlaciones de la distribución completa de {statistic} generacional ({experiment_name})",
        "box_title_template_est" : "Boxplots de {estimator} de {indicator} ({statistic} generacional)",
        "box_title_template_boot" : "Boxplots de {indicator} con bootstrap (n={n_boot}) ({statistic} generacional)",
        "box_title_template_all" : "Boxplots de la distribución completa de {indicator} ({statistic} generacional)",
        "violin_title_template_est" : "Violinplots de {estimator} de {indicator} ({statistic} generacional)",
        "violin_title_template_boot" : "Violinplots de {indicator} con bootstrap (n={n_boot}) ({statistic} generacional)",
        "violin_title_template_all" : "Violinplots de la distribución completa de {indicator} ({statistic} generacional)",
        "kde_title_template_est" : "KDEs de {estimator} de {indicator} ({statistic} generacional)",
        "kde_title_template_boot" : "KDEs de {indicator} con bootstrap (n={n_boot}) ({statistic} generacional)",
        "kde_title_template_all" : "KDEs de la distribución completa de {indicator} ({statistic} generacional)",
        "best" : "mejor",
        "worst" : "peor",
        "average" : "promedio",
        "std" : "std",
        "median" : "mediana",
        "Active Voxels" : "Voxeles activos",
        "Passive Voxels" : "Voxeles pasivos",
        "Generation" : "Generación"
    })
}

app = Flask(__name__)
CORS(app)
dal = Dal()
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": "Helvetica",
# })

def encode_image(img : Image):
    buffer = io.BytesIO()
    img.save(buffer, 'png')

    buffer.seek(0)
    
    data = buffer.read()
    data = base64.b64encode(data).decode()
    return data

def encode_image_lst(imgs):
    data_lst = []

    for img in imgs:
        data_lst += [encode_image(img)]

    return data_lst

def parse_lang(lang):
    return lang if lang is not None else 'en'

def boot_dist(data, n_boot=10000):
    boot_dist_li = []
    for _ in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist_li.append(np.mean(sample, axis=0))
    b = np.array(boot_dist_li)
    return b

def bootstrap(data, n_boot=10000, ci=68):
    b = boot_dist(data, n_boot=n_boot)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)
    
def tsplotboot(ax, data, ci = 68, n_boot = 10000, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = bootstrap(data, ci = ci, n_boot=n_boot)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est)
    ax.margins(x=0)

def tsplot(ax, data, **kw):
    x = np.arange(data.shape[0])
    ax.plot(x,data,**kw)
    ax.margins(x=0)

def max_size_run_statistics_ts(df, statistic, total_runs):
    run_statistic_mat = []
    max_length = -1
    for i in range(total_runs):
        run_i = df[df['run_number'] == i + 1]
        if len(run_i) > max_length:
            max_length = len(run_i)
            # reset mat
            run_statistic_mat = []
            run_statistic_mat += [run_i[statistic].tolist()]
        elif len(run_i) == max_length:
            # keep adding the ones that have the maximum length so far
            run_statistic_mat += [run_i[statistic].tolist()]
    run_statistic_mat = np.array(run_statistic_mat, dtype=np.float64)
    return run_statistic_mat

def validate_indicator_list(indicator_list):
    return all(indicator in INDICATOR_STATS_SET for indicator in indicator_list)

def validate_statistic_list(statistic_list):
    return all(statistic in STATISTICS for statistic in statistic_list)

def IndicatorBsConvergencePlots(indicators, statistics, population_type, experiment_names, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]

    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)

    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    all_experiments_stats = pd.DataFrame(all_experiments_stats)

    array_of_img_arrays = []
    for indicator in indicators:
        indicator_compact = compactify_indicator(indicator)
        img_array = []
        for statistic in statistics:
            if indicator in NO_STD_INDICATORS and statistic in ["worst","std","median","average"]:
                continue
            fig, ax = plt.subplots(ncols=1, sharey=True)
            for experiment_name in experiment_names:
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                if len(experiment_names) > 1:
                    tsplotboot(ax, run_statistic_mat, n_boot = n_boot, ci=95, label=experiment_name)
                    ax.legend()
                else:
                    tsplotboot(ax, run_statistic_mat, n_boot = n_boot, ci=95)
            ax.set_ylabel(f"{indicator_compact} ({lang_dict[statistic]})")
            ax.set_xlabel(lang_dict["Generation"])
            if len(experiment_names) > 1:
                ax.set_title(f"{indicator_compact} ({lang_dict[statistic]}), bootstrapping n={n_boot}")
            else:
                ax.set_title(f"{indicator_compact} ({lang_dict[statistic]}), bootstrapping n={n_boot}, {experiment_names[0]}")
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
            plt.close(fig)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

def IndicatorJointKdePlot(indicator1, indicator2, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]
    
    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)
    if not validate_indicator_list([indicator1, indicator2]):
        raise InvalidAPIUsage(f'No {indicator1}/{indicator2} indicator exists!', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child'] \
        and indicator1 not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"] \
        and indicator2 not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
        indicator1 = population_type + '_' + indicator1
        indicator2 = population_type + '_' + indicator2

    estimator_func = None
    if estimator is not None and estimator in STATISTICS:
        if estimator == "best":
            estimator_func = np.max
        elif estimator == "worst":
            estimator_func = np.min
        elif estimator == "average":
            estimator_func = np.mean
        elif estimator == "std":
            estimator_func = np.std
        elif estimator == "median":
            estimator_func = np.median
    elif estimator is not None and estimator not in STATISTICS: 
        raise InvalidAPIUsage(f'No {estimator} estimator supported!', status_code=404)

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    experiment_stats = dal.get_experiment_indicators_stats(run_ids, [indicator1, indicator2])
    if not experiment_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment and/or {indicator1} indicator!', status_code=404)
        
    df = pd.DataFrame(experiment_stats)
    compact_indicator1 = compactify_indicator(indicator1)
    compact_indicator2 = compactify_indicator(indicator2)
    img_array = []
    for statistic in statistics:
        df_list = []
        for experiment_name in experiment_names:
            df1 = df[(df['indicator'] == indicator1) & (df['experiment_name'] == experiment_name)]
            df2 = df[(df['indicator'] == indicator2) & (df['experiment_name'] == experiment_name)]
            if estimator_func:
                run_statistic_mat1 = max_size_run_statistics_ts(df1, statistic, exp_run_mapping[experiment_name])
                run_statistic_mat2 = max_size_run_statistics_ts(df2, statistic, exp_run_mapping[experiment_name])
                est1 = estimator_func(run_statistic_mat1, axis=0)
                est2 = estimator_func(run_statistic_mat2, axis=0)
                processed_data1 = est1
                processed_data2 = est2
            elif bootsrapped_dist:
                run_statistic_mat1 = max_size_run_statistics_ts(df1, statistic, exp_run_mapping[experiment_name])
                run_statistic_mat2 = max_size_run_statistics_ts(df2, statistic, exp_run_mapping[experiment_name])
                b1 = boot_dist(run_statistic_mat1, n_boot=n_boot)
                b2 = boot_dist(run_statistic_mat2, n_boot=n_boot)
                processed_data1 = b1.flatten()
                processed_data2= b2.flatten()
            else:
                processed_data1 = df1[statistic]
                processed_data2 = df2[statistic]
            new_df = {compact_indicator1 : processed_data1.astype('float').tolist(), compact_indicator2 : processed_data2.astype('float').tolist()}
            new_df = pd.DataFrame(new_df)
            new_df['experiment'] = experiment_name
            df_list += [new_df]
        
        resulting_df =  pd.concat(df_list, ignore_index=True)
        # fig,ax = plt.subplots(ncols=1)
        sns.set(style="darkgrid")
        g = sns.jointplot(data=resulting_df, x=compact_indicator1, y=compact_indicator2, kind= 'kde', hue='experiment', levels = 24, height=9, ratio=2)
        # g = sns.jointplot(data=resulting_df, x=indicator1, y=indicator2,  height=9, ratio=2)
        # g.plot_joint(sns.kdeplot, hue='experiment', zorder=0, levels=20)
        g.plot_marginals(sns.histplot, kde = True)
        if estimator:
            g.figure.suptitle(lang_dict["jointkde_title_template_est"].format(statistic=lang_dict[statistic], estimator=lang_dict[estimator]), y = 1.)
        elif bootsrapped_dist:
            g.figure.suptitle(lang_dict["jointkde_title_template_boot"].format(statistic=lang_dict[statistic], n_boot=lang_dict[n_boot]), y = 1.)
        else:
            g.figure.suptitle(lang_dict["jointkde_title_template_all"].format(statistic=lang_dict[statistic]), y = 1.)
        buffer = io.BytesIO()
        g.savefig(buffer, format='png')
        buffer.seek(0)
        data = buffer.read()
        data = base64.b64encode(data).decode()
        img_array += [data]
        plt.close(g.figure)
    return img_array

def IndicatorPairPlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]
    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)
    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator

    estimator_func = None
    if estimator is not None and estimator in STATISTICS:
        if estimator == "best":
            estimator_func = np.max
        elif estimator == "worst":
            estimator_func = np.min
        elif estimator == "average":
            estimator_func = np.mean
        elif estimator == "std":
            estimator_func = np.std
        elif estimator == "median":
            estimator_func = np.median
    elif estimator is not None and estimator not in STATISTICS: 
        raise InvalidAPIUsage(f'No {estimator} estimator supported!', status_code=404)

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    
    all_experiments_stats = pd.DataFrame(all_experiments_stats)
    
    pairplots_img_array = []
    correlations_img_array = []
    correlation_tables_array = []
    
    for statistic in statistics:
        df_list = []
        compacted_indicators = set()
        for experiment_name in experiment_names:
            new_df = {}
            for indicator in indicators:
                if indicator in NO_STD_INDICATORS and statistic in ["worst","std","median","average"]:
                    continue
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                if estimator_func:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    est = estimator_func(run_statistic_mat, axis=0)
                    processed_data = est.tolist()
                elif bootsrapped_dist:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    b = boot_dist(run_statistic_mat, n_boot=n_boot)
                    processed_data = b.flatten().tolist()
                else:
                    processed_data = df[statistic].tolist()
                compacted_indicators.add(compactify_indicator(indicator))
                new_df[compactify_indicator(indicator)] = processed_data
            new_df = pd.DataFrame(new_df)
            new_df['experiment'] = experiment_name
            df_list += [new_df]
        
        resulting_df = pd.concat(df_list, ignore_index=True)
        for indicator_compact in compacted_indicators:
            resulting_df[indicator_compact] = resulting_df[indicator_compact].astype('float')
        sns.set(style="darkgrid")
        g = sns.pairplot(resulting_df, hue="experiment", markers = ['o' for _ in experiment_names])
        g.map_lower(sns.regplot, scatter_kws = {'edgecolors' : [(1., 1., 1., 0.) for _ in experiment_names]})

        if estimator:
            g.figure.suptitle(lang_dict["pairplot_title_template_est"].format(statistic=lang_dict[statistic], estimator=lang_dict[estimator]), y=1.)
        elif bootsrapped_dist:
            g.figure.suptitle(lang_dict["pairplot_title_template_boot"].format(statistic=lang_dict[statistic], n_boot=lang_dict[n_boot]), y=1.)
        else:
            g.figure.suptitle(lang_dict["pairplot_title_template_all"].format(statistic=lang_dict[statistic]), y=1.)
        buffer = io.BytesIO()
        g.savefig(buffer, format='png')
        buffer.seek(0)
        data1 = buffer.read()
        data1 = base64.b64encode(data1).decode()
        pairplots_img_array += [data1]  
        plt.close(g.figure)
        corr_imgs = []
        corr_tables = []
        for experiment_name in experiment_names:
            experiment_indicator_data = resulting_df[resulting_df['experiment'] == experiment_name]
            correlation_table = experiment_indicator_data.corr()
            correlation_table_np = correlation_table.to_numpy()

            fig,ax = plt.subplots(figsize=correlation_table_np.shape)
            ax.grid(visible = False)
            cmap = mpl.colormaps['viridis']
            norm = mpl.colors.Normalize()
            pos = ax.imshow(correlation_table,norm=norm, cmap=cmap)
            columns_count = len(compacted_indicators)
            ax.set_xticks(range(columns_count), compacted_indicators, rotation=90)
            ax.set_yticks(range(columns_count), compacted_indicators, rotation=45)

            # Loop over data dimensions and create text annotations.
            for i in range(len(compacted_indicators)):
                for j in range(len(compacted_indicators)):
                    text = ax.text(j, i,  "{:.4f}".format(correlation_table.iloc[i, j]),
                                ha="center", va="center", color="w")

            fig.colorbar(pos, ax=ax, shrink=0.6, anchor=(0, 0.5))
            # ax.xticks(rotation=90)
            if estimator:
                ax.set_title(lang_dict["corr_title_template_est"].format(statistic=lang_dict[statistic], estimator=lang_dict[estimator], experiment_name = experiment_name), y=1.)
            elif bootsrapped_dist:
                ax.set_title(lang_dict["corr_title_template_boot"].format(statistic=lang_dict[statistic], n_boot=lang_dict[n_boot], experiment_name = experiment_name), y=1.)
            else:
                ax.set(title=lang_dict["corr_title_template_all"].format(statistic=lang_dict[statistic], experiment_name = experiment_name), y=1.)
            fig.tight_layout()
            fig.canvas.draw()
            data2 = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data2 = data2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            data2 = Image.fromarray(data2.astype('uint8')).convert('RGBA')
            data2 = encode_image(data2)
            corr_imgs += [data2]
            plt.close(fig)
            corr_tables += [correlation_table_np.tolist()]

        correlations_img_array.append(corr_imgs)
        correlation_tables_array.append(corr_tables)

    return pairplots_img_array, correlations_img_array, correlation_tables_array

def IndicatorBoxPlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]
    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)

    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator
    
    estimator_func = None
    if estimator is not None and estimator in STATISTICS:
        if estimator == "best":
            estimator_func = np.max
        elif estimator == "worst":
            estimator_func = np.min
        elif estimator == "average":
            estimator_func = np.mean
        elif estimator == "std":
            estimator_func = np.std
        elif estimator == "median":
            estimator_func = np.median
    elif estimator is not None and estimator not in STATISTICS: 
        raise InvalidAPIUsage(f'No {estimator} estimator supported!', status_code=404)

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    all_experiments_stats = pd.DataFrame(all_experiments_stats)
    
    array_of_img_arrays = []
    for indicator in indicators:
        img_array = []
        indicator_compact = compactify_indicator(indicator)
        for statistic in statistics:
            if indicator in NO_STD_INDICATORS and statistic in ["worst","std","median","average"]:
                continue
            df_list = []
            for experiment_name in experiment_names:
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                if estimator_func:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    est = estimator_func(run_statistic_mat, axis=0)
                    processed_data = est.tolist()
                elif bootsrapped_dist:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    b = boot_dist(run_statistic_mat, n_boot=n_boot)
                    processed_data = b.flatten().tolist()
                else:
                    processed_data = df[statistic].tolist()

                df_entry = {}
                df_entry[statistic] = processed_data
                df_entry = pd.DataFrame(df_entry)
                df_entry['experiment'] = experiment_name
                df_list += [df_entry]
        
            resulting_df =  pd.concat(df_list, ignore_index=True)
            resulting_df[indicator_compact] = resulting_df[statistic].astype('float')
            
            sns.set(style="darkgrid")
            fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(10,8))
            # g = sns.displot(data=resulting_df, x = indicator, hue='experiment', kind="kde", height=5, aspect=1.5)
            g = sns.boxplot(ax = ax, data=resulting_df, x = 'experiment', y=indicator_compact)
            if estimator:
                g.set(title=lang_dict["box_title_template_est"].format(statistic=lang_dict[statistic], indicator=indicator_compact, estimator=lang_dict[estimator]))
            elif bootsrapped_dist:
                g.set(title=lang_dict["box_title_template_boot"].format(statistic=lang_dict[statistic], indicator=indicator_compact, n_boot=lang_dict[n_boot]))
            else:
                g.set(title=lang_dict["box_title_template_all"].format(statistic=lang_dict[statistic], indicator=indicator_compact))

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
            plt.close(fig)
            plt.close(g.figure)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

def IndicatorViolinPlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]
    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)

    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator
    
    estimator_func = None
    if estimator is not None and estimator in STATISTICS:
        if estimator == "best":
            estimator_func = np.max
        elif estimator == "worst":
            estimator_func = np.min
        elif estimator == "average":
            estimator_func = np.mean
        elif estimator == "std":
            estimator_func = np.std
        elif estimator == "median":
            estimator_func = np.median
    elif estimator is not None and estimator not in STATISTICS: 
        raise InvalidAPIUsage(f'No {estimator} estimator supported!', status_code=404)

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    all_experiments_stats = pd.DataFrame(all_experiments_stats)
    
    array_of_img_arrays = []
    for indicator in indicators:
        img_array = []
        indicator_compact = compactify_indicator(indicator)
        for statistic in statistics:
            if indicator in NO_STD_INDICATORS and statistic in ["worst","std","median","average"]:
                continue
            df_list = []
            for experiment_name in experiment_names:
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                if estimator_func:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    est = estimator_func(run_statistic_mat, axis=0)
                    processed_data = est.tolist()
                elif bootsrapped_dist:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    b = boot_dist(run_statistic_mat, n_boot=n_boot)
                    processed_data = b.flatten().tolist()
                else:
                    processed_data = df[statistic].tolist()

                df_entry = {}
                df_entry[statistic] = processed_data
                df_entry = pd.DataFrame(df_entry)
                df_entry['experiment'] = experiment_name
                df_list += [df_entry]
        
            resulting_df =  pd.concat(df_list, ignore_index=True)
            resulting_df[indicator_compact] = resulting_df[statistic].astype('float')
            
            sns.set(style="darkgrid")
            fig, ax = plt.subplots(ncols=1, sharey=True, figsize=(11,9))
            # g = sns.displot(data=resulting_df, x = indicator, hue='experiment', kind="kde", height=5, aspect=1.5)
            g = sns.violinplot(ax = ax, data=resulting_df, x = 'experiment', y=indicator_compact)
            if estimator:
                g.set(title=lang_dict["violin_title_template_est"].format(statistic=lang_dict[statistic], indicator=indicator_compact, estimator=lang_dict[estimator]))
            elif bootsrapped_dist:
                g.set(title=lang_dict["violin_title_template_boot"].format(statistic=lang_dict[statistic], indicator=indicator_compact, n_boot=lang_dict[n_boot]))
            else:
                g.set(title=lang_dict["violin_title_template_all"].format(statistic=lang_dict[statistic], indicator=indicator_compact))

            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
            plt.close(fig)
            plt.close(g.figure)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

def IndicatorKdePlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000, lang = 'en'):
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict = LANG_DICTS[lang]
    if not validate_statistic_list(statistics):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)

    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator
    
    estimator_func = None
    if estimator is not None and estimator in STATISTICS:
        if estimator == "best":
            estimator_func = np.max
        elif estimator == "worst":
            estimator_func = np.min
        elif estimator == "average":
            estimator_func = np.mean
        elif estimator == "std":
            estimator_func = np.std
        elif estimator == "median":
            estimator_func = np.median
    elif estimator is not None and estimator not in STATISTICS: 
        raise InvalidAPIUsage(f'No {estimator} estimator supported!', status_code=404)

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    all_experiments_stats = pd.DataFrame(all_experiments_stats)
    
    array_of_img_arrays = []
    for indicator in indicators:
        img_array = []
        indicator_compact = compactify_indicator(indicator)
        for statistic in statistics:
            if indicator in NO_STD_INDICATORS and statistic in ["worst","std","median","average"]:
                continue
            df_list = []
            for experiment_name in experiment_names:
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                if estimator_func:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    est = estimator_func(run_statistic_mat, axis=0)
                    processed_data = est.tolist()
                elif bootsrapped_dist:
                    run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                    b = boot_dist(run_statistic_mat, n_boot=n_boot)
                    processed_data = b.flatten().tolist()
                else:
                    processed_data = df[statistic].tolist()

                df_entry = {}
                df_entry[statistic] = processed_data
                df_entry = pd.DataFrame(df_entry)
                df_entry['experiment'] = experiment_name
                df_list += [df_entry]
        
            resulting_df =  pd.concat(df_list, ignore_index=True)
            resulting_df[indicator_compact] = resulting_df[statistic].astype('float')

            sns.set(style="darkgrid")
            # g = sns.displot(data=resulting_df, x = indicator, hue='experiment', kind="kde", height=5, aspect=1.5)
            g = sns.displot(data=resulting_df, x = indicator_compact, hue='experiment', kde=True, height=8, aspect=1.2)
            if estimator:
                g.set(title=lang_dict["kde_title_template_est"].format(statistic=lang_dict[statistic], indicator=indicator_compact, estimator=lang_dict[estimator]))
            elif bootsrapped_dist:
                g.set(title=lang_dict["kde_title_template_boot"].format(statistic=lang_dict[statistic], indicator=indicator_compact, n_boot=lang_dict[n_boot]))
            else:
                g.set(title=lang_dict["kde_title_template_all"].format(statistic=lang_dict[statistic], indicator=indicator_compact))

            buffer = io.BytesIO()
            g.savefig(buffer, format='png')
            buffer.seek(0)
            data = buffer.read()
            data = base64.b64encode(data).decode()
            img_array += [data]
            plt.close(g.figure)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

def StructuredArchivePlots(archive, indicator, statistic, experiment_names, lang = 'en'):
    indicator2Indx = {"fitness" : 2, "unaligned_novelty" : 3, "aligned_novelty" : 4}
    if not lang in LANG_DICTS:
        raise InvalidAPIUsage(f'Please specify a valid language choice as query string', status_code=404)
    lang_dict : Dict[str, str]= LANG_DICTS[lang]

    if archive not in ARCHIVES:
        raise InvalidAPIUsage(f'No {archive} archive exists!', status_code=404)
    if statistic not in STATISTICS:
        raise InvalidAPIUsage(f'No {statistic} statistic exists!', status_code=404)

    run_ids = {}
    experiment_parameters = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)
        experiment_parameters[experiment_name] = experiment_obj['parameters']
        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids[experiment_name] = experiment_runs["run_id"]
    
    experiment_archives = {}
    for experiment_name in experiment_names:
        run_archive = dal.get_archives_json(run_ids[experiment_name])
        if not run_archive["archives_json"]:
            raise InvalidAPIUsage(f'No archive data from runs available for {experiment_name} experiment!', status_code=404)
        if archive not in run_archive["archives_json"][0]:
            raise InvalidAPIUsage(f'No archive {archive} exists!', status_code=404)
        experiment_archives[experiment_name] = [archive_json[archive] for archive_json in run_archive["archives_json"]]
        
        for i in range(len(experiment_archives[experiment_name])):
            processed_archive = []
            run_archive = experiment_archives[experiment_name][i]
            for elem in run_archive:
                if type(elem) is list:
                    processed_archive += [elem[indicator2Indx[indicator]]]
                else:
                    processed_archive += [0.]
            experiment_archives[experiment_name][i] = processed_archive
        experiment_archives[experiment_name] = np.array(experiment_archives[experiment_name])
    
    img_array = []
    for experiment_name in experiment_names:
        
        if statistic == "best":
            flattened_feat_map = np.max(experiment_archives[experiment_name], axis = 0)
        elif statistic == "worst":
            flattened_feat_map = np.min(experiment_archives[experiment_name], axis = 0)
        elif statistic == "average":
            flattened_feat_map = np.mean(experiment_archives[experiment_name], axis = 0)
        elif statistic == "std":
            flattened_feat_map = np.std(experiment_archives[experiment_name], axis = 0)
        elif statistic == "median":
            flattened_feat_map = np.median(experiment_archives[experiment_name], axis = 0)

        parameters = experiment_parameters[experiment_name]
        total_voxels = 1
        for voxel in parameters["IND_SIZE"]:
            total_voxels *= voxel
        if archive[:4] == "f_me":
            x_points = parameters["me_evaluator"]["bpd"][0]
            y_points = parameters["me_evaluator"]["bpd"][1]
        elif archive[:5] == "an_me":
            x_points = parameters["an_me_evaluator"]["bpd"][0]
            y_points = parameters["an_me_evaluator"]["bpd"][1]

        feats = [list(np.linspace(0, total_voxels, x_points)), list(np.linspace(0, total_voxels, y_points))]
        bc_space = []
        for element in itertools.product(*feats):
            bc_space += [list(element)]
        bc_space = np.array(bc_space)

        # xreg = np.linspace(0, total_voxels, total_voxels)
        # yreg = np.linspace(0, total_voxels, total_voxels)
        xreg = np.linspace(0, total_voxels, x_points)
        yreg = np.linspace(0, total_voxels, y_points)
        X,Y = np.meshgrid(xreg,yreg)
        fig, (ax) = plt.subplots(ncols=1, sharey=True, figsize=(10,8))
        ax.set_xlabel(lang_dict['Active Voxels'])
        ax.set_ylabel(lang_dict['Passive Voxels'])
        ax.set_title(lang_dict["archives_title_template"].format(statistic=lang_dict[statistic], indicator=lang_dict[indicator], experiment_name = experiment_name))
        x, y, z = bc_space[:,0], bc_space[:,1], flattened_feat_map
        Z = spinterp.griddata(np.vstack((x,y)).T,z,(X,Y),
                      method='linear').reshape(X.shape)
        col = ax.pcolormesh(X,Y,Z.T)
        fig.colorbar(col, ax=ax, location='right')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
        plt.close(fig)
    return img_array

def ChooseWinner(indicators, statistic, population_type, experiment_names):
    if not validate_statistic_list([statistic]):
        raise InvalidAPIUsage(f'Please specify a set of valid statistics to plot as query string', status_code=404)
    if not validate_indicator_list(indicators):
        raise InvalidAPIUsage(f'Please specify a set of valid indicators to plot as query string', status_code=404)
    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)
    pop_prefix = ''
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'
    
    if pop_prefix != '':
        for j, indicator in enumerate(indicators):
            if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
                indicators[j] = pop_prefix + indicator

    run_ids = []
    exp_run_mapping = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids += experiment_runs["run_id"]
        exp_run_mapping[experiment_name] = len(experiment_runs["run_id"])

    if len(indicators) < len(INDICATOR_STATS_SET):
        all_experiments_stats= dal.get_experiment_indicators_stats(run_ids, indicators)
    else:
        indicators = INDICATOR_STATS_SET
        all_experiments_stats= dal.get_experiment_stats(run_ids)
    if not all_experiments_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    all_experiments_stats = pd.DataFrame(all_experiments_stats)

    # Compute indicator scores per algo
    alpha = 0.05
    indicator_algo_scores = dict(zip(indicators, [{exp : 0 for exp in experiment_names} for _ in indicators]))
    condorcet_scores = dict(zip(experiment_names, [{exp : 0 for exp in experiment_names} for _ in experiment_names]))
    permutation_counts = {}
    indicator_algo_tables = {}
    for indicator in indicators:
        indicator_algo_tables[indicator] = {}
        for i in range(len(experiment_names)):
            algo1 = experiment_names[i]
            df1 = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == algo1)]
            data1 = df1[statistic].to_numpy()
            mu1 = data1.mean()
            sigma1 = data1.std()
            indicator_algo_tables[indicator][algo1] = ((mu1, sigma1), {})
            for j in range(i + 1, len(experiment_names)):
                algo2 = experiment_names[j]
                df2 = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == algo2)]
                data2 = df2[statistic].to_numpy()
                mu2 = data2.mean()
                sigma2 = data2.std()
                res = stats.wilcoxon(data1, y = data2, correction=True)
                if res.pvalue < alpha:
                    # significantly different
                    if mu1 > mu2:
                        # algo1 is superior to algo2 in regards to indicator
                        condorcet_scores[algo1][algo2] += 1
                        indicator_algo_scores[indicator][algo1]+=1
                        indicator_algo_scores[indicator][algo2]-=1
                        indicator_algo_tables[indicator][algo1][1][algo2] = (mu2, sigma2, res.pvalue, 2)
                    else:
                        # algo2 is superior to algo1 in regards to indicator
                        condorcet_scores[algo2][algo1] += 1
                        indicator_algo_scores[indicator][algo1]-=1
                        indicator_algo_scores[indicator][algo2]+=1
                        indicator_algo_tables[indicator][algo1][1][algo2] = (mu2, sigma2, res.pvalue, 1)
                else:
                    # couldn't reject null hypothesis
                    indicator_algo_tables[indicator][algo1][1][algo2] = (mu2, sigma2, res.pvalue, 0)
        # We now sort the algos based on their results for the current indicator
        sorted_algos = dict(sorted(indicator_algo_scores[indicator].items(), key = lambda x : x[1], reverse=True))
        permutation = tuple(sorted_algos.keys())
        # Next we add up to the permutation count, this will be useful when computing the borda count for each algo
        if permutation in permutation_counts:
            permutation_counts[permutation] += 1
        else:
            permutation_counts[permutation] = 1
    
    # We now count the number of times each algo was 1st, 2nd, ..., etc.
    places_count = {exp : [0]*len(experiment_names) for exp in experiment_names}
    for permutation, count in permutation_counts.items():
        for i, algo in enumerate(permutation):
            places_count[algo][i] += count

    # We now get the borda count for each algo, given each place count
    borda_count = {}
    place_weights = np.arange(len(experiment_names), 0, -1)
    for exp in experiment_names:
        places_count_vec = np.array(places_count[exp])
        borda_count[exp] = int(np.dot(places_count_vec, place_weights))

    borda_count = dict(sorted(borda_count.items(), key = lambda x : x[1], reverse=True))

    return condorcet_scores, borda_count, places_count, {",".join(perm) : count for perm, count in permutation_counts.items()}, indicator_algo_scores, indicator_algo_tables

class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        return rv

@app.errorhandler(InvalidAPIUsage)
def invalid_api_usage(e):
    return jsonify(e.to_dict()), e.status_code

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/IndicatorPairPlotsRender/mode/<mode>", methods = ["GET"])
def IndicatorPairPlotsRenderGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        paiplot_img_array, corr_img_array, _ = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            paiplot_img_array, corr_img_array, _ = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            paiplot_img_array, corr_img_array, _ = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            paiplot_img_array, corr_img_array, _ = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            paiplot_img_array, corr_img_array, _ = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "<br>".join([f'<img src="data:image/png;base64,{pairplot_img}">' + "".join([f'<img src="data:image/png;base64,{corr_img}">' for corr_img in corr_imgs]) for pairplot_img, corr_imgs in zip(paiplot_img_array, corr_img_array)])

@app.route("/IndicatorPairPlots/mode/<mode>", methods = ["GET"])
def IndicatorPairPlotsGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        paiplot_img_array, corr_img_array, corr_table_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            paiplot_img_array, corr_img_array, corr_table_array  = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            paiplot_img_array, corr_img_array, corr_table_array  = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            paiplot_img_array, corr_img_array, corr_table_array  = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            paiplot_img_array, corr_img_array, corr_table_array  = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'pairplot_imgs': paiplot_img_array,
            'corr_imgs' : corr_img_array,
            'corr_tables' : corr_table_array
        })

@app.route("/IndicatorJointKdePlotRender/indicator1/<indicator1>/indicator2/<indicator2>/mode/<mode>", methods = ["GET"])
def IndicatorJointKdePlotRenderGET(indicator1, indicator2, mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "".join([f'<img src="data:image/png;base64,{img}">' for img in img_array])

@app.route("/IndicatorJointKdePlot/indicator1/<indicator1>/indicator2/<indicator2>/mode/<mode>", methods = ["GET"])
def IndicatorJointKdePlotGET(indicator1, indicator2, mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'img': img_array
        })

@app.route("/IndicatorKdePlotsRender/mode/<mode>", methods = ["GET"])
def IndicatorKdePlotsRenderGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "<br>".join(["".join([f'<img src="data:image/png;base64,{img}">' for img in img_array]) for img_array in array_of_img_arrays])

@app.route("/IndicatorKdePlots/mode/<mode>", methods = ["GET"])
def IndicatorKdePlotsGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'img': array_of_img_arrays
        })

@app.route("/IndicatorBoxPlotsRender/mode/<mode>", methods = ["GET"])
def IndicatorBoxPlotsRenderGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "<br>".join(["".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array]) for img_array in array_of_img_arrays])

@app.route("/IndicatorBoxPlots/mode/<mode>", methods = ["GET"])
def IndicatorBoxPlotsGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorBoxPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'size': [[[img.width, img.height] for img in img_array] for img_array in array_of_img_arrays], 
            'format': array_of_img_arrays[0][0].format,
            'img': [[encode_image(img) for img in img_array] for img_array in array_of_img_arrays]
        })

@app.route("/IndicatorViolinPlotsRender/mode/<mode>", methods = ["GET"])
def IndicatorViolinPlotsRenderGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "<br>".join(["".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array]) for img_array in array_of_img_arrays])

@app.route("/IndicatorViolinPlots/mode/<mode>", methods = ["GET"])
def IndicatorViolinPlotsGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot, lang=lang)
        else:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, lang=lang)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator, lang=lang)
        else:
            array_of_img_arrays = IndicatorViolinPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average', lang=lang)
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'size': [[[img.width, img.height] for img in img_array] for img_array in array_of_img_arrays], 
            'format': array_of_img_arrays[0][0].format,
            'img': [[encode_image(img) for img in img_array] for img_array in array_of_img_arrays]
        })

@app.route("/IndicatorBsConvergencePlotsRender/n_boot/<n_boot>", methods = ["GET"])
def IndicatorBsConvergencePlotsRenderGET(n_boot):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if n_boot == 'default':
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    else:
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, n_boot=n_boot, lang=lang)

    return "<br>".join(["".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array]) for img_array in array_of_img_arrays])

@app.route("/IndicatorBsConvergencePlots/n_boot/<n_boot>", methods = ["GET"])
def IndicatorBsConvergencePlotsGET(n_boot):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    lang = parse_lang(args.get('lang'))
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if n_boot == 'default':
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, lang=lang)
    else:
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, n_boot=n_boot, lang=lang)
    return jsonify({
            'msg': 'success', 
            'size': [[[img.width, img.height] for img in img_array] for img_array in array_of_img_arrays], 
            'format': array_of_img_arrays[0][0].format,
            'img': [[encode_image(img) for img in img_array] for img_array in array_of_img_arrays]
        })

@app.route("/IndicatorPlotRunRender/experiment/<experiment_name>/indicator/<indicator>/run/<run_number>/statistic/<statistic>", methods = ["GET"])
def IndicatorPlotRunRenderGET(experiment_name, indicator, run_number, statistic):
    args = request.args
    population_type = args.get('population')

    if indicator not in INDICATOR_STATS_SET:
        raise InvalidAPIUsage(f'No {indicator} indicator exists!', status_code=404)
    if statistic not in STATISTICS:
        raise InvalidAPIUsage(f'No {statistic} statistic for {indicator} exists!', status_code=404)

    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child'] and indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
        indicator = population_type + '_' + indicator

    experiment_obj = dal.get_experiment(experiment_name)
    if not experiment_obj:
        raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

    experiment_run = dal.get_experiment_run(experiment_obj["experiment_id"], run_number)
    if not experiment_run["run_id"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    
    run_id = experiment_run["run_id"]
    experiment_stats = dal.get_experiment_indicator_run_stats(run_id, indicator)
    if not experiment_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    df = pd.DataFrame(experiment_stats)
    run_statistic = df[statistic].to_numpy()

    fig, (ax) = plt.subplots(ncols=1, sharey=True)
    tsplot(ax, run_statistic)
    ax.set_ylabel(f"{indicator} {statistic}")
    ax.set_xlabel("Generation")
    ax.set_title(f"{indicator} {statistic} for {experiment_name} experiment, run {run_number}")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')

    return f'<img src="data:image/png;base64,{encode_image(img)}">'

@app.route("/AllPlotRunRender/experiment/<experiment_name>/run/<run_number>/statistic/<statistic>", methods = ["GET"])
def AllPlotRunRenderGET(experiment_name, run_number, statistic):
    args = request.args
    population_type = args.get('population')
    pop_prefix = ''

    if statistic not in STATISTICS:
        raise InvalidAPIUsage(f'No {statistic} statistic exists!', status_code=404)

    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child'] and indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
        pop_prefix = population_type + '_'

    experiment_obj = dal.get_experiment(experiment_name)
    if not experiment_obj:
        raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

    experiment_run = dal.get_experiment_run(experiment_obj["experiment_id"], run_number)
    if not experiment_run["run_id"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    
    run_id = experiment_run["run_id"]
    experiment_stats = dal.get_experiment_run_stats(run_id)
    if not experiment_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    df = pd.DataFrame(experiment_stats)

    img_array = []
    for indicator in INDICATOR_STATS_SET:
        if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
            indicator = pop_prefix + indicator
        run_statistic = df[df['indicator'] == indicator]
        run_statistic = run_statistic[statistic].to_numpy()

        fig, (ax) = plt.subplots(ncols=1, sharey=True)
        tsplot(ax, run_statistic)
        ax.set_ylabel(f"{indicator} {statistic}")
        ax.set_xlabel("Generation")
        ax.set_title(f"{indicator} {statistic} for {experiment_name} experiment, run {run_number}")
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]


    return "".join([f'<img src="data:image/png;base64,{encode_image(img)}"><br>' for img in img_array])

@app.route("/StructuredArchivePlotsRender/archive/<archive>/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def StructuredArchivePlotsRenderGET(archive, indicator, statistic):
    args = request.args
    experiment_names = args.getlist('experiments')
    lang = parse_lang(args.get('lang'))
    img_array = StructuredArchivePlots(archive, indicator, statistic, experiment_names, lang=lang)
    return "".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array])

@app.route("/StructuredArchivePlots/archive/<archive>/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def StructuredArchivePlotsGET(archive, indicator, statistic):
    args = request.args
    experiment_names = args.getlist('experiments')
    lang = parse_lang(args.get('lang'))
    img_array = StructuredArchivePlots(archive, indicator, statistic, experiment_names, lang=lang)
    return jsonify({
        'msg': 'success', 
        'size': [[img.width, img.height] for img in img_array], 
        'format': img_array[0].format,
        'img': [encode_image(img) for img in img_array]
    })

@app.route("/ChooseWinner/statistic/<statistic>", methods = ["GET"])
def ChooseWinnerGET(statistic):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    indicator_list = WINNING_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    condorcet_scores, borda_count, places_count, permutation_counts, indicator_algo_scores, indicator_algo_tables = ChooseWinner(indicator_list, statistic, population_type, experiment_names)
    return jsonify({
            'msg': 'success', 
            'condorcet_scores' : condorcet_scores,
            'borda_count': borda_count, 
            'places_count': places_count,
            'permutation_counts': permutation_counts,
            'indicator_algo_scores': indicator_algo_scores,
            'indicator_algo_tables': indicator_algo_tables
        })

if __name__ == '__main__':
    app.run(debug=True)
    