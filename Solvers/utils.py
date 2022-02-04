from timeit import default_timer

def stop_process_pool(executor):
    for pid, process in executor._processes.items():
        process.terminate()
    executor.shutdown()

def time_remaining(start,time_limit):
    return max(time_limit - (default_timer()-start),0)