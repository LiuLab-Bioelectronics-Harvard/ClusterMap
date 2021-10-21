"""
scanpy related utilites.

"""

# Import packages
import os
import numpy as np
import pandas as pd
import textwrap as tw
from anndata import AnnData
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import ttest_ind, norm, ranksums, spearmanr
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from scipy.stats.mstats import zscore


def show_stat(adata):
    no_gene = np.sum(adata.var["total_counts"] == 0)
    no_read = np.sum(adata.obs["total_counts"] == 0)
    batches = adata.obs.batch.unique()

    count_dict = {}
    for i in batches:
        count_dict[i] = sum(adata.obs['batch'] == i)

    message = tw.dedent(f"""\
     Number of cells: {adata.X.shape[0]}
     Cells without gene: {no_gene}
     Cells without reads: {no_read}
     Number of genes: {adata.X.shape[1]}
     Number of batches: {len(batches)}
     Cells in each experiment: {count_dict}
     """)
    print(message)


def show_reads_quantile(adata):
    read_quantile = adata.obs['total_counts'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print(f"Reads per cell quantile:\n{read_quantile}")


# Plot per cell stats plot
def plot_stats_per_cell(adata, color='sample', save=False):
    plt.figure(figsize=(15, 5))

    reads_per_cell = adata.obs['total_counts']
    genes_per_cell = adata.obs['n_genes_by_counts']

    plt.subplot(1, 3, 1)
    sns.histplot(reads_per_cell)
    plt.ylabel('# cells')
    plt.xlabel('# reads')

    plt.subplot(1, 3, 2)
    sns.histplot(genes_per_cell)
    plt.ylabel('# cells')
    plt.xlabel('# genes')

    plt.subplot(1, 3, 3)
    plt.title(
        'R=%f' % np.corrcoef(reads_per_cell.T, genes_per_cell)[0, 1])  # Pearson product-moment correlation coefficients
    sns.scatterplot(data=adata.obs, x='total_counts', y='n_genes_by_counts', hue=color, s=5)
    plt.xlabel("Reads per cell")
    plt.ylabel("Genes per cell")
    plt.legend(loc='lower right')
    plt.tight_layout()

    if save:
        # current_fig_path = os.path.join(os.getcwd(), "output/figures/cell_stats.pdf")
        current_fig_path = "./output/figures/cell_stats.pdf"
        plt.savefig(current_fig_path)
    plt.show()


# Filter by cell area
def filter_cells_by_area(adata, min_area, max_area, save=False):

    # Plot cell area distribution
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(adata.obs['area'])

    # Filter by cell area
    adata = adata[adata.obs.area > min_area, :]
    adata = adata[adata.obs.area < max_area, :]

    # Plot cell area distribution after filtration
    plt.subplot(1, 2, 2)
    sns.histplot(adata.obs['area'])

    plt.tight_layout()

    if save:
        current_fig_path = "./output/figures/cell_filter_by_area.pdf"
        plt.savefig(current_fig_path)

    plt.show()
    print(f"Number of cell left: {adata.X.shape[0]}")

    return adata





# Plot heatmap of gene markers of each cluster
def plot_heatmap_with_labels(adata, degenes, cluster_key, cmap=plt.cm.get_cmap('tab10'), show_axis=True,
                             show_top_ticks=True, use_labels=None, font_size=15, annotation=None):

    g = plt.GridSpec(2, 1, wspace=0.01, hspace=0.01, height_ratios=[0.5, 10])
    cluster_vector = adata.obs[cluster_key].values.astype(int)
    cluster_array = np.expand_dims(np.sort(cluster_vector), 1).T
    ax = plt.subplot(g[0])
    ax.imshow(cluster_array, aspect='auto', interpolation='none', cmap=cmap)
    if show_top_ticks:
        locations = []
        for i in np.unique(cluster_vector):
            locs = np.median(np.argwhere(cluster_array == i)[:, 1].flatten())
            locations.append(locs)
        ax.xaxis.tick_top()
        if use_labels is not None:
            plt.xticks(locations, use_labels)
        else:
            plt.xticks(locations, np.unique(cluster_vector))
        ax.get_yaxis().set_visible(False)
    # ax.axis('off')

    ax = plt.subplot(g[1])
    plot_heatmap(adata, list(degenes), cluster_key, fontsize=font_size, use_imshow=False, ax=ax, annotation=annotation)
    if not show_axis:
        plt.axis('off')


# Plot heatmap
def plot_heatmap(adata, gene_names, cluster_key, cmap=plt.cm.get_cmap('bwr'), fontsize=16,
                 use_imshow=False, ax=None, show_vlines=True, annotation=None):

    input_data = adata.X
    cluster_vector = adata.obs[cluster_key].values.astype(int)

    if ax is None:
        ax = plt.axes()

    clust_sizes = [sum(cluster_vector == i) for i in np.unique(cluster_vector)]
    # data = np.vstack([input_data.iloc[cluster_vector == i, :].loc[:, gene_names].values for i in np.unique(cluster_vector)]).T
    gene_index = []
    for v in gene_names:
        curr_index = np.where(adata.var.index.to_numpy() == v)[0]
        if len(curr_index) != 0:
            gene_index.append(curr_index[0])
        else:
            raise Exception(f"Gene: {v} not included!")

    data = np.vstack(
        [input_data[cluster_vector == i, :][:, gene_index] for i in np.unique(cluster_vector)]).T


    if use_imshow:
        ax.imshow(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap, interpolation='none', aspect='auto')
    else:
        ax.pcolor(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap)
    # plt.imshow(np.flipud(zscore(data,axis=1)),vmin=-2.5, vmax=2.5, cmap=cmap, aspect='auto', interpolation='none')
    ax.set_xlim([0, data.shape[1]])
    ax.set_ylim([0, data.shape[0]])
    if show_vlines:
        for i in np.cumsum(clust_sizes[:-1]):
            ax.axvline(i, color='k', linestyle='-')
    ax.set_yticks(np.arange(data.shape[0])+0.5)

    if not annotation:
        y_labels = gene_names[::-1]
    else:
        y_labels = []
        for gene in gene_names[::-1]:
            y_labels.append("%s - %s" % (gene, annotation[gene]))

    ax.set_yticklabels(y_labels, fontsize=fontsize)
    # ax.get_xaxis().set_fontsize(fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)


def merge_multiple_clusters(adata, clusts, cluster_key='louvain'):
    cluster_vector = adata.obs[cluster_key].values.astype(int)
    for idx, c in enumerate(clusts[1:]):
        cluster_vector[cluster_vector == c] = clusts[0]
    temp = cluster_vector.copy()
    # relabel clusters to be contiguous
    for idx, c in enumerate(np.unique(cluster_vector)):
        temp[cluster_vector == c] = idx
    adata.obs[cluster_key] = temp
    adata.obs[cluster_key] = adata.obs[cluster_key].astype('category')


# Get Convexhull for each cell
def get_qhulls(labels):
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(regionprops(labels)):
        hulls.append(ConvexHull(region.coords))
        coords.append(region.coords)
        centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")
    return hulls, coords, centroids


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster_by_sample(adata, sample, cmap, save=False, show=True,
                            width=2, height=9, figscale=10,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):
    sample = f"{sample}_morph"
    nissl = adata.uns[sample]['label_img']
    hulls = adata.uns[sample]['qhulls']
    colors = adata.uns[sample]['colors']
    good_cells = adata.uns[sample]['good_cells']

    plt.figure(figsize=(figscale*width/float(height), figscale))
    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        # others = [p for i, p in enumerate(polys) if i not in good_cells]
        polys = [p for i, p in enumerate(polys) if i in good_cells]

    p = PatchCollection(polys, alpha=alpha, cmap=cmap, edgecolor='k', linewidth=0.5)
    # o = PatchCollection(others, alpha=0.1, cmap=other_cmap, edgecolor='k', linewidth=0.5)

    if vmin or vmax is not None:
        p.set_array(colors)
        p.set_clim(vmin=vmin, vmax=vmax)
    else:
        if rescale_colors:
            p.set_array(colors+1)
            p.set_clim(vmin=0, vmax=max(colors+1))
        else:
            p.set_array(colors)
            p.set_clim(vmin=0, vmax=max(colors))

    # show background image (nissl | DAPI | device)
    nissl = (nissl > 0).astype(np.int)
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.gca().add_collection(p)
    # plt.gca().add_collection(o)
    plt.axis('off')
    plt.tight_layout()

    if save:
        if isinstance(save, str):
            current_fig_path = f"./output/figures/{save}_{sample}_cell_type.pdf"
            plt.savefig(current_fig_path)
        else:
            current_fig_path = f"./output/figures/{sample}_cell_type.pdf"
            plt.savefig(current_fig_path)

    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()


# Convert convexhull to polygon
def hull_to_polygon(hull):
    cent = np.mean(hull.points, 0)
    pts = []
    for pt in hull.points[hull.simplices]:
        pts.append(pt[0].tolist())
        pts.append(pt[1].tolist())
    pts.sort(key=lambda p: np.arctan2(p[1] - cent[1],
                                      p[0] - cent[0]))
    pts = pts[0::2]  # Deleting duplicates
    pts.insert(len(pts), pts[0])
    k = 1.1
    poly = Polygon(k * (np.array(pts) - cent) + cent, edgecolor='k', linewidth=1)
    # poly.set_capstyle('round')
    return poly