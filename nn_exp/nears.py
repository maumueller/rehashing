#!/usr/bin/env python3

import numpy as np
import time
import gzip
import os
import sys
import random
from numba import jit


def compute_contrib(data, queries, b):
    contribs = np.zeros((queries.shape[0],data.shape[0]))
    for i, q in enumerate(queries):
        print(f'computing kde values for query vector {i+1}/{queries.shape[0]}')
        m = data - q
        contribs[i] = np.exp(-np.linalg.norm(data_points - q, axis=1)/b)
    return contribs


if __name__ == '__main__':

    fn = sys.argv[1]
    qn = int(sys.argv[2])
    b = 1.0
    if len(sys.argv) > 3:
        b = float(sys.argv[3])


    data = np.loadtxt(fn, delimiter=",")
    data = data[:, 1:]
    (n,d) = data.shape

    queries = random.choices(list(range(n)), k=qn)

    data_points = np.array([x for i, x in enumerate(data) if not i in queries])
    query_points = np.array([x for i, x in enumerate(data) if i in queries])

    contribs = compute_contrib(data_points, query_points, b)

    print('performing cumsum arithmetic')
    np.flip(contribs,axis=1).sort()
    contribs = contribs[:,1:]
    contribs /= np.sum(contribs, axis=1)[:,None]
    contribs = np.cumsum(contribs, axis=1)
    print('saving array')
    with gzip.open(f'{fn}_b_{b}.npy.gz','w') as f:
        np.save(f,contribs)
    print('done.')

