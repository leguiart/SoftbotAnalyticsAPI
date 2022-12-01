
import numpy as np
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

from data.dal import Dal

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
    
def tsplotboot(ax, data,ci = 68, **kw):
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = bootstrap(data, ci = ci)
    ax.fill_between(x,cis[0],cis[1],alpha=0.2, **kw)
    ax.plot(x,est,**kw)
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

@app.route("/BootstrappedIndicatorPlotGET/experiment/<experiment_name>/indicator/<indicator>/statistic/<statistic>", methods = ["GET"])
def ImageBatchGET(experiment_name, indicator, statistic):
    experiment_obj = dal.get_experiment(experiment_name)
    if not experiment_obj:
        raise InvalidAPIUsage(f'No experiment named {experiment_name} exists!', status_code=404)

    experiment_runs = dal.get_experiment_runs(experiment_obj["experiment_id"])
    if not experiment_runs["run_id"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
    
    run_ids = experiment_runs["run_id"]
    experiment_stats = dal.get_experiment_indicator_stats(run_ids, indicator)
    if not experiment_stats["best"]:
        raise InvalidAPIUsage(f'No data from runs available for {experiment_name} experiment!', status_code=404)
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

    fig, (ax) = plt.subplots(ncols=1, sharey=True)
    tsplotboot(ax, run_statistic_mat, ci=95)
    ax.set_ylabel(f"{indicator} {statistic}")
    ax.set_xlabel("Generation")
    ax.set_title(f"Bootstrapped {indicator} {statistic} for {experiment_name} experiment")
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    imarray = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(imarray.astype('uint8')).convert('RGBA')

    # return jsonify({
    #         'msg': 'success', 
    #         'size': [img.width, img.height], 
    #         'format': img.format,
    #         'img': encoded_img
    #     })
    return f'<img src="data:image/png;base64,{encode_image(img)}">'


if __name__ == '__main__':
    app.run(debug=True)
    