
import itertools
import numpy as np
import io
import base64
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp

from scipy import stats
from PIL import Image
from flask_cors import CORS
from flask import Flask, jsonify, request

# from application.data.dal import Dal # debug
from data.dal import Dal

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

ARCHIVES = ["f_me_archive",
            "an_me_archive",
            "novelty_archive_un",
            "novelty_archive_an"]

STATISTICS = ["best",
              "worst",
              "average",
              "std",
              "median"]

app = Flask(__name__)
CORS(app)
dal = Dal()

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

def IndicatorBsConvergencePlots(indicators, statistics, population_type, experiment_names, n_boot = 10000):
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
        img_array = []
        for statistic in statistics:
            fig, ax = plt.subplots(ncols=1, sharey=True)
            for experiment_name in experiment_names:
                df = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == experiment_name)]
                run_statistic_mat = max_size_run_statistics_ts(df, statistic, exp_run_mapping[experiment_name])
                if len(experiment_names) > 1:
                    tsplotboot(ax, run_statistic_mat, n_boot = n_boot, ci=95, label=experiment_name)
                    ax.legend()
                else:
                    tsplotboot(ax, run_statistic_mat, n_boot = n_boot, ci=95)
            ax.set_ylabel(f"{indicator} {statistic}")
            ax.set_xlabel("Generation")
            if len(experiment_names) > 1:
                ax.set_title(f"Bootstrapped {indicator} {statistic}")
            else:
                ax.set_title(f"Bootstrapped {indicator} {statistic} for {experiment_names[0]} experiment")
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
            plt.close(fig)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

def IndicatorJointKdePlot(indicator1, indicator2, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000):
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
                processed_data1 = est1.tolist()
                processed_data2 = est2.tolist()
            elif bootsrapped_dist:
                run_statistic_mat1 = max_size_run_statistics_ts(df1, statistic, exp_run_mapping[experiment_name])
                run_statistic_mat2 = max_size_run_statistics_ts(df2, statistic, exp_run_mapping[experiment_name])
                b1 = boot_dist(run_statistic_mat1, n_boot=n_boot)
                b2 = boot_dist(run_statistic_mat2, n_boot=n_boot)
                processed_data1 = b1.flatten().tolist()
                processed_data2= b2.flatten().tolist()
            else:
                processed_data1 = df1[statistic].tolist()
                processed_data2 = df2[statistic].tolist()
            new_df = {indicator1 : processed_data1, indicator2 : processed_data2}
            new_df = pd.DataFrame(new_df)
            new_df['experiment'] = experiment_name
            df_list += [new_df]
        
        resulting_df =  pd.concat(df_list, ignore_index=True)
        # fig,ax = plt.subplots(ncols=1)
        sns.set(style="darkgrid")
        g = sns.jointplot(data=resulting_df, x=indicator1, y=indicator2, kind= 'kde', hue='experiment', levels = 24, height=9, ratio=2)
        # g.plot_joint(sns.kdeplot, zorder=0, levels=20)
        # g.plot_marginals(sns.kdeplot)
        if estimator:
            g.figure.suptitle(f'Joint KDE plots of {estimator} of generational {statistic}')
        elif bootsrapped_dist:
            g.figure.suptitle(f'Joint KDE plots of bootstrapped (n={n_boot}) generational {statistic}')
        else:
            g.figure.suptitle(f'Joint KDE plots of full generational {statistic} distribution')
        buffer = io.BytesIO()
        g.savefig(buffer, format='png')
        buffer.seek(0)
        data = buffer.read()
        data = base64.b64encode(data).decode()
        img_array += [data]
        plt.close(g.figure)
    return img_array

def IndicatorPairPlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000):
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
    img_array = []
    for statistic in statistics:
        df_list = []
        for experiment_name in experiment_names:
            new_df = {}
            for indicator in indicators:
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
                new_df[indicator] = processed_data
            new_df = pd.DataFrame(new_df)
            new_df['experiment'] = experiment_name
            df_list += [new_df]
        
        resulting_df =  pd.concat(df_list, ignore_index=True)
        # fig,ax = plt.subplots(ncols=1)
        sns.set(style="darkgrid")
        g = sns.pairplot(resulting_df, hue="experiment", markers = ['o' for _ in experiment_names])
        g.map_lower(sns.regplot, scatter_kws = {'edgecolors' : [(1., 1., 1., 0.) for _ in experiment_names]})
        if estimator:
            g.figure.suptitle(f'Regression/Scatter plots of {estimator} of generational {statistic}')
        elif bootsrapped_dist:
            g.figure.suptitle(f'Regression/Scatter plots of bootstrapped (n={n_boot}) generational {statistic}')
        else:
            g.figure.suptitle(f'Regression/Scatter plots of full generational {statistic} distribution')
        
        buffer = io.BytesIO()
        g.savefig(buffer, format='png')
        buffer.seek(0)
        data = buffer.read()
        data = base64.b64encode(data).decode()
        img_array += [data]
        plt.close(g.figure)
    return img_array

def IndicatorKdePlots(indicators, statistics, population_type, experiment_names, estimator = None, bootsrapped_dist = False, n_boot = 10000):
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
        for statistic in statistics:
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
            resulting_df[indicator] = resulting_df[statistic]

            sns.set(style="darkgrid")
            g = sns.displot(data=resulting_df, x = indicator, hue='experiment', kind="kde", height=5, aspect=1.5)
            if estimator:
                g.set(title=f'KDE plot of {estimator} of generational {indicator} {statistic}')
            elif bootsrapped_dist:
                g.set(title=f'KDE plot of bootstrapped (n={n_boot}) generational {indicator} {statistic}')
            else:
                g.set(title=f'KDE plot of full generational {indicator} {statistic} distribution')

            buffer = io.BytesIO()
            g.savefig(buffer, format='png')
            buffer.seek(0)
            data = buffer.read()
            data = base64.b64encode(data).decode()
            img_array += [data]
            plt.close(g.figure)
        array_of_img_arrays.append(img_array)
    return array_of_img_arrays

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
    for indicator in indicators:
        for i in range(len(experiment_names)):
            algo1 = experiment_names[i]
            df1 = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == algo1)]
            data1 = df1[statistic].to_numpy()
            mu1 = data1.mean()
            # sigma1 = data1.std()
            for j in range(i + 1, len(experiment_names)):
                algo2 = experiment_names[j]
                df2 = all_experiments_stats[(all_experiments_stats['indicator'] == indicator) & (all_experiments_stats['experiment_name'] == algo2)]
                data2 = df2[statistic].to_numpy()
                mu2 = data2.mean()
                # sigma2 = data2.std()
                res = stats.wilcoxon(data1, y = data2, correction=True)
                if res.pvalue < alpha:
                    # significantly different
                    if mu1 > mu2:
                        condorcet_scores[algo1][algo2] += 1
                        indicator_algo_scores[indicator][algo1]+=1
                        indicator_algo_scores[indicator][algo2]-=1
                    else:
                        condorcet_scores[algo2][algo1] += 1
                        indicator_algo_scores[indicator][algo1]-=1
                        indicator_algo_scores[indicator][algo2]+=1
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

    return condorcet_scores, borda_count, places_count, {",".join(perm) : count for perm, count in permutation_counts.items()}, indicator_algo_scores

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
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average')
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "".join([f'<img src="data:image/png;base64,{img}">' for img in img_array])

@app.route("/IndicatorPairPlots/mode/<mode>", methods = ["GET"])
def IndicatorPairPlotsGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            img_array = IndicatorPairPlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average')
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'size': [[img.width, img.height] for img in img_array], 
            'format': img_array[0].format,
            'img': img_array
        })

@app.route("/IndicatorJointKdePlotRender/indicator1/<indicator1>/indicator2/<indicator2>/mode/<mode>", methods = ["GET"])
def IndicatorJointKdePlotRenderGET(indicator1, indicator2, mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    statistic_list = args.getlist('statistics')
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names)
    if mode == 'full':
        img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator='average')
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return "".join([f'<img src="data:image/png;base64,{img}">' for img in img_array])

@app.route("/IndicatorJointKdePlot/indicator1/<indicator1>/indicator2/<indicator2>/mode/<mode>", methods = ["GET"])
def IndicatorJointKdePlotGET(indicator1, indicator2, mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    statistic_list = args.getlist('statistics')
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names)
    if mode == 'full':
        img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            img_array = IndicatorJointKdePlot(indicator1, indicator2, statistic_list, population_type, experiment_names, estimator='average')
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'size': [[img.width, img.height] for img in img_array], 
            'format': img_array[0].format,
            'img': img_array
        })

@app.route("/IndicatorKdePlotsRender/mode/<mode>", methods = ["GET"])
def IndicatorKdePlotsRenderGET(mode):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average')
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
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if mode == 'full':
        array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names)
    elif mode == 'bootstrap_dist':
        n_boot = args.get('n_boot')
        if n_boot:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True, n_boot=n_boot)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, bootsrapped_dist=True)
    elif mode == 'est':
        estimator = args.get('estimator')
        if estimator:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator=estimator)
        else:
            array_of_img_arrays = IndicatorKdePlots(indicator_list, statistic_list, population_type, experiment_names, estimator='average')
    else:
        raise InvalidAPIUsage(f'Mode {mode} is not valid!', status_code=404)
    return jsonify({
            'msg': 'success', 
            'size': [[[img.width, img.height] for img in img_array] for img_array in array_of_img_arrays], 
            'format': array_of_img_arrays[0][0].format,
            'img': array_of_img_arrays
        })

@app.route("/IndicatorBsConvergencePlotsRender/n_boot/<n_boot>", methods = ["GET"])
def IndicatorBsConvergencePlotsRenderGET(n_boot):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if n_boot == 'default':
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names)
    else:
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, n_boot=n_boot)

    return "<br>".join(["".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array]) for img_array in array_of_img_arrays])

@app.route("/IndicatorBsConvergencePlots/n_boot/<n_boot>", methods = ["GET"])
def IndicatorBsConvergencePlotsGET(n_boot):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    statistic_list = args.getlist('statistics')
    indicator_list = PLOT_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    statistic_list = STATISTICS if len(statistic_list) == 1 and statistic_list[0] == 'all' else statistic_list
    if n_boot == 'default':
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names)
    else:
        array_of_img_arrays = IndicatorBsConvergencePlots(indicator_list, statistic_list, population_type, experiment_names, n_boot=n_boot)
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

@app.route("/ArchivesPlotRender/archive/<archive>/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def ArchivesPlotRenderGET(archive, indicator, statistic):
    args = request.args
    experiment_names = args.getlist('experiments')
    indicator2Indx = {"fitness" : 2, "unaligned_novelty" : 3, "aligned_novelty" : 4}

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

        xreg = np.linspace(0, total_voxels, total_voxels)
        yreg = np.linspace(0, total_voxels, total_voxels)
        X,Y = np.meshgrid(xreg,yreg)
        fig, (ax) = plt.subplots(ncols=1, sharey=True)
        ax.set_xlabel('Active Voxels')
        ax.set_ylabel('Passive Voxels')
        ax.set_title(f"{indicator} {statistic} feature map for {experiment_name} experiment")

        x, y, z = bc_space[:,0], bc_space[:,1], flattened_feat_map
        Z = spinterp.griddata(np.vstack((x,y)).T,z,(X,Y),
                      method='linear').reshape(X.shape)
        col = ax.pcolormesh(X,Y,Z.T)
        fig.colorbar(col, ax=ax, location='right')
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img_array += [Image.fromarray(imarray.astype('uint8')).convert('RGBA')]
    
    return "".join([f'<img src="data:image/png;base64,{encode_image(img)}">' for img in img_array])

@app.route("/ChooseWinner/statistic/<statistic>", methods = ["GET"])
def ChooseWinnerGET(statistic):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    indicator_list = args.getlist('indicators')
    indicator_list = WINNING_INDICATORS.copy() if len(indicator_list) == 1 and indicator_list[0] == 'all' else indicator_list
    condorcet_scores, borda_count, places_count, permutation_counts, indicator_algo_scores = ChooseWinner(indicator_list, statistic, population_type, experiment_names)
    return jsonify({
            'msg': 'success', 
            'condorcet_scores' : condorcet_scores,
            'borda_count': borda_count, 
            'places_count': places_count,
            'permutation_counts': permutation_counts,
            'indicator_algo_scores': indicator_algo_scores
        })

if __name__ == '__main__':
    app.run(debug=True)
    