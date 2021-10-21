"""
This file contains all basic utility functions.
"""

# Import packages
import numpy as np
from scipy.stats import chi2, norm


# Test package loading
def test():
    print('hello world xxx')


# Get the index of a subset
def get_subset_index(p, i):
    return np.argwhere(p.values == i).flatten()


# Calculate differential gene expression lrt
def differential_lrt(x, y):  # removed argument xmin=0
    lrt_x = bimod_likelihood(x)
    lrt_y = bimod_likelihood(y)
    lrt_z = bimod_likelihood(np.concatenate((x, y)))
    lrt_diff = 2 * (lrt_x + lrt_y - lrt_z)
    return chi2.pdf(x=lrt_diff, df=3)


# Change the range of an array
def min_max(x, x_min, x_max):
    x[x > x_max] = x_max
    x[x < x_min] = x_min
    return x


# Calculate binomial bimod_likelihood
def bimod_likelihood(x, xmin=0):
    x1 = x[x <= xmin]
    x2 = x[x > xmin]
    # xal = MinMax(x2.shape[0]/x.shape[0], 1e-5, 1-1e-5)
    xal = len(x2) / float(len(x))
    if xal < 1e-5:
        xal = 1e-5
    if xal > 1 - 1e-5:
        xal = 1 - 1e-5
    lik_a = x1.shape[0] * np.log(1 - xal)
    if len(x2) < 2:
        mysd = 1
    else:
        mysd = np.std(x2)
    lik_b = x2.shape[0] * np.log(xal) + np.sum(np.log(norm.pdf(x2, np.mean(x2), mysd)))
    return lik_a + lik_b


# Convert rc based coordinates to xy
def rc2xy_2d(rc):
    x = rc[:, 1]
    r_max = max(rc[:, 0])
    y = r_max - rc[:, 0] + 1
    xy = np.column_stack((x, y))
    return xy


# Convert rcz based coordinates to xyz
def rc2xy_3d(rcz):
    temp = rcz.copy()
    x = temp[:, 1]
    r_max = max(temp[:, 0])
    y = r_max - temp[:, 0] + 1
    z = temp[:, 2]
    xyz = np.column_stack((x, y, z))
    return xyz


# Merge multiple clusters in AnnData object
def merge_multiple_clusters_ann(adata, clusts):
    org_clusts = adata.obs['leiden'].astype(int).to_numpy()
    for idx, c in enumerate(clusts[1:]):
        org_clusts[org_clusts == c] = clusts[0]
    temp = org_clusts.copy()
    # relabel clusters to be contiguous
    for idx, c in enumerate(np.unique(org_clusts)):
        temp[org_clusts == c] = idx
    adata.obs['leiden'] = temp.astype(str)
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')


# Sort clusters with input order
def order_clusters_ann(adata, orders):
    clusts = adata.obs['leiden'].astype(int).to_numpy()
    temp = clusts.copy()
    n_clusters = len(np.unique(clusts))
    for idx, c in enumerate(np.unique(clusts)):
        temp[clusts == c] = idx + n_clusters

    for idx, c in enumerate(np.unique(temp)):
        temp[temp == c] = orders[idx]
        print(f"{idx} --> {orders[idx]}")
    adata.obs['leiden'] = temp.astype(str)
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
