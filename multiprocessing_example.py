# import multiprocessing
# import time
# from datetime import datetime

# def worker_function(x):
#     '''Simuliert Rechenintensive Aufgabe'''
#     time.sleep(0.5)
#     return x * x

# if __name__ == "__main__":
#     start_time = datetime.now()
#     inputs = range(10)

#     with multiprocessing.Pool(processes=3) as pool:
#         results = pool.map(worker_function, inputs)

#     end_time = datetime.now()

#     # comparison with single process time
#     single_process_time_start = datetime.now()
#     for i in inputs:
#         worker_function(i)
#     single_process_time_end = datetime.now()

#     print()
#     print("Results:", results)
#     print("Multi process time:", (end_time - start_time).total_seconds(), "seconds")
#     print("Single process time:", (single_process_time_end - single_process_time_start).total_seconds(), "seconds")
#     print()

import multiprocessing
import time

def worker_function(x):
    '''Simuliert Rechenintensive Aufgabe'''
    time.sleep(0.5)
    return x * x

if __name__ == "__main__":
    inputs = range(10)

    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(worker_function, inputs)