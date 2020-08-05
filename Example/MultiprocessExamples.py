import multiprocessing as mp
import numpy as np
import datetime as dt
import pandas as pd
import HpDispatcher

from tqdm import tqdm

import pickle


np.random.RandomState(100)
arr = np.random.randint(0, 10, size=[50, 2])
data = arr.tolist()


def howmany_within_range(row, i=0, minimum=4, maximum=8):
    """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
    count = 0
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return i, count


def benchmarking_scalar_product(vec0, vec1=None, index=0, iterations=100):
    """Returns the scalar product between two vectors, computes it for a given number of iterations"""
    if vec1 is None:
        vec1 = vec0
    for _ in range(0, iterations):
        np.dot(np.array(vec0), np.array(vec1))
    return index, np.dot(np.array(vec0), np.array(vec1))


def time_resulsts(results, msg='', previous=None):
    """Returns timestamp for function performance analysis Helper function"""
    time_stamp = dt.datetime.today()

    if previous is None:
        print(msg, ' ' * (40 - len(msg)), time_stamp, results[:10] * (len(results) > 0))
    else:
        print(msg, ' ' * (40 - len(msg)), time_stamp, results[:10] * (len(results) > 0),
              f'delta_t: {time_stamp - previous}')

    return time_stamp


results = []

time_stamp = time_resulsts(results, 'Start')

for row in data:
    results.append(benchmarking_scalar_product(row))

time_stamp = time_resulsts(results, 'end for loop', time_stamp)

pool = mp.Pool(mp.cpu_count())

# results = [pool.apply(benchmarking_scalar_product, args=[row]) for row in data]
#
# time_stamp = time_resulsts(results, 'end pool.apply', time_stamp)

results = pool.map(benchmarking_scalar_product, [row for row in data])

time_stamp = time_resulsts(results, 'end pool.map', time_stamp)

results = pool.starmap(benchmarking_scalar_product, [(row, row, 0, 100 )for row in data])

time_stamp = time_resulsts(results, 'end pool.starmap', time_stamp)

results = pool.imap(benchmarking_scalar_product, [row for row in data])
results = [res for res in results]

time_stamp = time_resulsts(results, 'end pool.imap', time_stamp)

# results_objects = [pool.apply_async(benchmarking_scalar_product, args=(row, row, i, 100)) for i, row in enumerate(data)]
#
# results = [r.get() for r in results_objects]
#
# time_stamp = time_resulsts(results, 'end pool.apply_async', time_stamp)

# results = pool.starmap_async(benchmarking_scalar_product, [(row, row, i, 100) for i, row in enumerate(data)]).get()
#
# time_stamp = time_resulsts(results, 'end pool.starmap_async', time_stamp)

pool.close()


def hypothenus(row):
    return round(row[0] ** 2 + row[1] ** 2, 2) ** 0.5


vec_size = 100
vec = np.array([np.linspace(1, vec_size, vec_size), np.linspace(1, vec_size, vec_size)]).T
df = pd.DataFrame(vec)
dispatch = HpDispatcher.HpDispatcher()
# df = pd.DataFrame(np.random.randint(3, 10, size=[500, 2]))

pool = mp.Pool(4)

time_stamp = time_resulsts(results, 'Start test 2')

results = pool.imap(hypothenus, df.itertuples(name=None, index=False), chunksize=10)
output = [round(x, 2) for x in results]

time_stamp = time_resulsts(output, 'end pool.imap(df)', time_stamp)

results = pool.imap(hypothenus, vec, chunksize=10)
output = [round(x, 2) for x in results]

time_stamp = time_resulsts(output, 'end pool.imap(vec)', time_stamp)

pool.close()

# params = {row:vec}
dispatch.imap_iterable(hypothenus, vec)
time_stamp = time_resulsts(output, 'dispatch.map_iterable', time_stamp)