import ot
import numpy as np


def compute_emd(x, y, p):
    """
    Takes datasets x, y, and returns Wasserstein-p distance between them
    Inputs
    - x; source data. dimensions bs x d
    - y; target data. dimensions bs x d
    - p; power for cost matrix. 1 or 2
    Outputs
    - G0; permutation matrix representing optimal pairing between x and y
    - cost; transport cost from x to y.
    """
    # assumes datasets are equal size
    n = x.shape[0]

    a, b = np.ones((n,)) / n, np.ones((n,)) / n  # uniform distribution on samples

    # loss matrix
    M = ot.dist(x, y, metric='euclidean' if p == 1 else 'sqeuclidean')
    maxx = M.max()
    M /= M.max()

    G0 = ot.emd(a, b, M, log=True)
    G0, cost = G0[0], G0[1]['cost']

    G0 *= n  # G0 is a permutation matrix

    if G0.max(axis=0).sum() != n:
        print(G0.max(axis=0).sum())
        print('Warning: OT map does not exist')

    cost *= maxx

    return G0, cost
