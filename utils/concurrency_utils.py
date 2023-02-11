
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

def generic_om_parallel_execution(func, params, desired_parallelism, end_process_msg):
    results_list = []
    with ProcessPoolExecutor(max_workers=min(desired_parallelism,61)) as executor:
        for result in executor.map(func, params):
            print(end_process_msg)
            results_list += [result]
    return results_list

def generic_om_concurrent_execution(func, params, desired_concurrency, end_thread_msg):
    results_list = []
    cpu_count = os.cpu_count()
    max_threads = int(cpu_count) * 5 if cpu_count is not None else 5
    with ThreadPoolExecutor(max_workers=min(desired_concurrency, max_threads)) as executor:
        for result in executor.map(func, params):
            print(end_thread_msg)
            results_list += [result]
    return results_list

def generic_mm_parallel_execution(func_params, desired_parallelism, end_process_msg):
    results_list = []
    with ProcessPoolExecutor(max_workers=min(desired_parallelism,61)) as executor:
        futures = []
        for func, f_args, f_kwargs in func_params:
            futures += [executor.submit(func, *f_args, **f_kwargs)]
        for result in as_completed(futures):
            print(end_process_msg)
            results_list += [result]
    return results_list

def generic_mm_concurrent_execution(func_params, desired_concurrency, end_thread_msg):
    results_list = []
    cpu_count = os.cpu_count()
    max_threads = int(cpu_count) * 5 if cpu_count is not None else 5
    with ThreadPoolExecutor(max_workers=min(desired_concurrency, max_threads)) as executor:
        futures = []
        futures = [executor.submit(func, *f_args, **f_kwargs) for func, f_args, f_kwargs in func_params]
        for future in as_completed(futures):
            print(end_thread_msg)
            results_list += [future.result()]
    return results_list