
import base64
import time
import traceback
import json
import sys
import os
import logging
import argparse

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

    if os.path.exists(func_info_path):
        with open(func_info_path, "r") as outfile:
            func_info = json.load(outfile)
    else:
        logger.error(f'No file {func_info_path} exists')
        raise ValueError(f'No file {func_info_path} exists')

    
    func_params = json_to_func_params( img_base_path, host_url, delay, func_info)
    if _concurrent:
        retrieved_plots = generic_mm_concurrent_execution(func_params, p_num, 'Finished retrieving plot')
    else:
        retrieved_plots = generic_mm_parallel_execution(func_params, p_num, 'Finished retrieving plot')
    json_object = json.dumps(retrieved_plots, indent=4)
    with open("retrieved_plots.json", "w") as outfile:
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
    main(arg_parser)