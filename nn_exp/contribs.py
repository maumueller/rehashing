#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import gzip
import sys
from bisect import bisect_right

fn = sys.argv[1]

if __name__ == '__main__':
    with gzip.open(fn,'r') as f:
        contribs = np.load(f)

    (m,n) = contribs.shape

    hist_50 = [0]*m
    hist_90 = [0]*m
    hist_99 = [0]*m

    for i in range(m):
        if np.isnan(contribs[i,:]).any():
            continue
        hist_50[i] = bisect_right(contribs[i], 0.5)
        hist_90[i] = bisect_right(contribs[i], 0.9)
        hist_99[i] = bisect_right(contribs[i], 0.99)

    fig, axs = plt.subplots(3)
    fig.suptitle('# nearest neighbors that contribute >50/90/99% of the value')
    plt.xlabel('# nearest neighbors neighbors')
    plt.ylabel('query vector count')
    axs[0].hist(hist_50,bins=range(min(hist_50),max(hist_50)+1, 100))
    axs[1].hist(hist_90,bins=range(min(hist_90),max(hist_90)+1, 100))
    axs[2].hist(hist_99,bins=range(min(hist_99),max(hist_99)+1, 100))

    plt.savefig(f'{fn}_contribs.png')

