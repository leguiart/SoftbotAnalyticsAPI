
import itertools
import numpy as np
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spinterp
from scipy import stats
from PIL import Image
from flask_cors import CORS
from flask import Flask, jsonify, request, send_file


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

def bootstrap(data, n_boot=10000, ci=68):
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.-ci/2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50.+ci/2.)
    return (s1,s2)
    
def tsplotboot(ax, data, ci = 68, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = bootstrap(data, ci = ci)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est)
    ax.margins(x=0)

def tsplot(ax, data, **kw):
    x = np.arange(data.shape[0])
    ax.plot(x,data,**kw)
    ax.margins(x=0)

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

def BootstrappedIndicatorPlot(indicator, statistic, population_type, experiment_names):

    if indicator not in INDICATOR_STATS_SET:
        raise InvalidAPIUsage(f'No {indicator} indicator exists!', status_code=404)
    if statistic not in STATISTICS:
        raise InvalidAPIUsage(f'No {statistic} statistic for {indicator} exists!', status_code=404)

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)
    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child'] and indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
        indicator = population_type + '_' + indicator

    fig, (ax) = plt.subplots(ncols=1, sharey=True)
    for experiment_name in experiment_names:

        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        
        run_ids = experiment_runs["run_id"]
        experiment_stats = dal.get_experiment_indicator_stats(run_ids, indicator)
        if not experiment_stats["best"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment and/or {indicator} indicator!', status_code=404)
        df = pd.DataFrame(experiment_stats)

        run_statistic_mat = []
        max_length = -1

        for i in range(len(run_ids)):
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
        if len(experiment_names) > 1:
            tsplotboot(ax, run_statistic_mat, ci=95, label=experiment_name)
            ax.legend()
        else:
            tsplotboot(ax, run_statistic_mat, ci=95)

    ax.set_ylabel(f"{indicator} {statistic}")
    ax.set_xlabel("Generation")
    if len(experiment_names) > 1:
        ax.set_title(f"Bootstrapped {indicator} {statistic}")
    else:
        ax.set_title(f"Bootstrapped {indicator} {statistic} for {experiment_names[0]} experiment")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
    return img

@app.route("/BootstrappedIndicatorPlotRender/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def BootstrappedIndicatorPlotRenderGET(indicator, statistic):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')

    img = BootstrappedIndicatorPlot(indicator, statistic, population_type, experiment_names)
    return f'<img src="data:image/png;base64,{encode_image(img)}">'

@app.route("/BootstrappedIndicatorPlot/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def BootstrappedIndicatorPlotGET(indicator, statistic):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')

    img = BootstrappedIndicatorPlot(indicator, statistic, population_type, experiment_names)
    return jsonify({
            'msg': 'success', 
            'size': [img.width, img.height], 
            'format': img.format,
            'img': encode_image(img)
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

@app.route("/BootstrappedAllPlotRender/statistic/<statistic>", methods = ["GET"])
def BootstrappedAllPlotRenderGET(statistic):
    args = request.args
    population_type = args.get('population')
    experiment_names = args.getlist('experiments')
    pop_prefix = ''

    if experiment_names is None: 
        raise InvalidAPIUsage(f'Please specify a set of experiments to plot as query string', status_code=404)

    if population_type and population_type not in ['parent', 'child']:
        raise InvalidAPIUsage(f'No {population_type} population type exists!', status_code=404)
    elif population_type and population_type in ['parent', 'child']:
        pop_prefix = population_type + '_'

    run_ids = {}
    for experiment_name in experiment_names:
        experiment_obj = dal.get_experiment(experiment_name)
        if not experiment_obj:
            raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

        experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
        if not experiment_runs["run_id"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        run_ids[experiment_name] = experiment_runs["run_id"]

    experiment_stats = {}
    for experiment_name in experiment_names:
        experiment_stats[experiment_name] = dal.get_experiment_stats(run_ids[experiment_name])
        if not experiment_stats[experiment_name]["best"]:
            raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
        experiment_stats[experiment_name] = pd.DataFrame(experiment_stats[experiment_name])

    img_array = []
    for indicator in INDICATOR_STATS_SET:
        if indicator not in ["qd-score_ff","qd-score_fun","qd-score_fan","qd-score_anf","qd-score_anun","qd-score_anan","coverage"]:
            indicator = pop_prefix + indicator
        fig, (ax) = plt.subplots(ncols=1, sharey=True)
        for experiment_name in experiment_names:

            run_statistic_mat = []
            max_length = -1
            df = experiment_stats[experiment_name]
            for i in range(len(run_ids[experiment_name])):
                
                run_i = df[(df['indicator'] == indicator) & (df['run_number'] == i + 1)]
                if len(run_i) > max_length:
                    max_length = len(run_i)
                    # reset mat
                    run_statistic_mat = []
                    run_statistic_mat += [run_i[statistic].tolist()]
                elif len(run_i) == max_length:
                    # keep adding the ones that have the maximum length so far
                    run_statistic_mat += [run_i[statistic].tolist()]
            run_statistic_mat = np.array(run_statistic_mat, dtype=np.float64)
            if len(experiment_names) > 1:
                tsplotboot(ax, run_statistic_mat, ci=95, label=experiment_name)
                ax.legend()
            else:
                tsplotboot(ax, run_statistic_mat, ci=95)
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


    return "".join([f'<img src="data:image/png;base64,{encode_image(img)}"><br>' for img in img_array])

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

        xreg = np.linspace(0, total_voxels, x_points)
        yreg = np.linspace(0, total_voxels, y_points)
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

if __name__ == '__main__':
    app.run(debug=True)
    