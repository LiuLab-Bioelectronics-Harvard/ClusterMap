"""
This file contains fuctions for visualization.
"""

# Import packages
from . import utilities as ut

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.stats import ttest_ind, norm, ranksums, spearmanr
from scipy.spatial import ConvexHull
from skimage.measure import regionprops
from scipy.stats.mstats import zscore
from anndata import AnnData


# ==== STATS ====
# Plot per cell stats plot
def plot_stats_per_cell(self):
    plt.figure(figsize=(15, 10))

    if isinstance(self, AnnData):
        reads_per_cell = self.obs['total_counts']
        genes_per_cell = self.obs['n_genes_by_counts']
        color = self.obs['orig_ident']
    else:
        reads_per_cell = self._meta["reads_per_cell"]
        genes_per_cell = self._meta["genes_per_cell"]
        color = self._meta["orig_ident"]

    plt.subplot(2, 3, 1)
    plt.hist(reads_per_cell, 10, color='k')
    plt.ylabel('# cells')
    plt.xlabel('# reads')

    plt.subplot(2, 3, 2)
    plt.hist(genes_per_cell, 10, color='k')
    plt.ylabel('# cells')
    plt.xlabel('# genes')

    plt.subplot(2, 3, 3)
    plt.title(
        'R=%f' % np.corrcoef(reads_per_cell.T, genes_per_cell)[0, 1])  # Pearson product-moment correlation coefficients
    plt.scatter(reads_per_cell, genes_per_cell, marker='.', s=30, c=color, cmap=plt.cm.get_cmap('jet'), lw=0)
    plt.xlabel("Reads per cell")
    plt.ylabel("Genes per cell")


# Plot correlation between pair of experiments
def plot_correlation_pair(self, exp_id_0, exp_id_1):
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    rep1 = np.sum(self.get_cells_by_experiment(idx=exp_id_0, use_scaled=False), axis=0) + 1
    rep2 = np.sum(self.get_cells_by_experiment(idx=exp_id_1, use_scaled=False), axis=0) + 1

    plt.title('Spearman R=%.4f' % spearmanr(rep1, rep2).correlation)
    plt.scatter(rep1, rep2, c="black")
    plt.xlabel("log2(rep1 + 1)")
    plt.ylabel("log2(rep2 + 1)")
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=2)
    plt.show()


# ==== DIMENSIONALITY REDUCTION ====
# Plot explained variances of PCA
def plot_explained_variance(self):
    if self._pca is not None:
        plt.plot(self._pca.explained_variance_ratio_, 'ko-')


# Plot PCA
def plot_pca(self, use_cells=None, dim0=0, dim1=1, colorby="orig_ident", s=10, cmap=plt.cm.get_cmap('jet')):
    if self._active_cells is None:
        plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
                         c=self.get_metadata(colorby), cmap=cmap, s=s, lw=0)
    else:
        new_color = self.get_metadata(colorby).iloc[self._active_cells]
        plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
                    c=new_color, cmap=cmap, s=s, lw=0)
    # if self._clusts is None:
    #     plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
    #                 c=self.get_metadata("orig_ident"), cmap=cmap, s=s, lw=0)
    # else:
    #     plt.scatter(self._transformed_pca[:, dim0], self._transformed_pca[:, dim1],
    #                 c=self._clusts, cmap=cmap, s=s, lw=0)


# Plot dimensionality reduction results (tsne/umap)
def plot_dim(self, cmap=None, colorby=None, s=10, renumber_clusts=False):
    """
    :param self:
    :param cmap: color map
    :param colorby
    :param s: dot size
    :param renumber_clusts:
    :return:
    """
    if self._clusts is None:
        plt.plot(self._tsne[:, 0], self._tsne[:, 1], 'o')
    elif colorby is not None:
        c = self._meta[colorby].cat.codes.to_list()
        plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=c, s=s, cmap=cmap, lw=0)
    else:
        if cmap is None:
            # get color map based on number of clusters
            cmap = plt.cm.get_cmap('jet', len(np.unique(self._clusts)))
        if not renumber_clusts:
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._clusts, s=s, cmap=cmap, lw=0)
        else:
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._clusts+1,
                        s=s, cmap=cmap, vmin=0, vmax=self._clusts.max() + 1, lw=0)
        plt.title(f"Number of clusters: {len(np.unique(self._clusts))}")


# Plot dimensionality reduction results of a specific sample
def plot_dim_org_id(self):
    plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=self._meta['orig_ident'], s=10, cmap=plt.cm.get_cmap('gist_rainbow'), lw=0)


# === EXPRESSION ====
# Plot expression between groups for sepecifc genes for all clusters
def plot_expression_between_groups(self, gene_names, test="bimod", plot_type="bar",
                                   figsize=(10,10), vmin=0, vmax=None, use_raw=False):
    # convert gene_names to list
    if not isinstance(gene_names, list):
        gene_names = [gene_names]

    group_vals = np.unique(self._meta["group"].values)
    # cluster_df = []
    # ncells = []

    f, ax = plt.subplots(nrows=len(gene_names), ncols=len(np.unique(self._clusts)), figsize=figsize)
    f.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)

    for i, g in enumerate(gene_names):
        for j, c in enumerate(np.unique(self._clusts)):
            cells = self.get_cells_by_cluster(c, use_raw=use_raw)

            # normalize to TPM
            meta = self.get_metadata_by_cluster(c)
            cells0 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[0]), :]
            cells1 = cells.iloc[ut.get_subset_index(meta["group"], group_vals[1]), :]
            n0 = cells0.shape[0]
            n1 = cells1.shape[0]
            expr = np.hstack((cells0[g].values, cells1[g].values))

            ids = np.hstack((np.zeros((n0,)), np.ones((n1,))))
            temp = np.zeros_like(ids)
            d = pd.DataFrame(data=np.vstack((expr, ids)).T, columns=["expr", "group"])

            if len(gene_names) == 1:
                curr_ax = ax[j]
            else:
                curr_ax = ax[i][j]

            if plot_type is "bar":
                sns.barplot(x="group", y="expr", data=d, ax=curr_ax, capsize=.2, errwidth=1)
            elif plot_type is "violin":
                sns.violinplot(x="group", y="expr", data=d, ax=curr_ax, capsize=.2, errwidth=1,
                               palette="Set2_r", inner=None, linewidth=0)
                sns.swarmplot(x="group", y="expr", data=d, ax=curr_ax, size=4, color=".3", linewidth=0)

            if test is "bimod":
                pval = ut.differential_lrt(cells0[g].values, cells1[g].values)
            if test is "wilcox":
                pval = ranksums(cells0[g].values, cells1[g].values)[1]
            if test is "t":
                pval = ttest_ind(cells0[g].values, cells1[g].values)[1]

            if vmax is not None:
                curr_ax.set_ylim([vmin, vmax])

            curr_ax.set_title("Gene %s\nCluster %d\nP=%2.4f" % (g, c, pval))
            curr_ax.get_xaxis().set_visible(False)
            sns.despine(fig=f, ax=curr_ax, bottom=True, left=True)
            # sns.violinplo


# Plot volcano plot
def plot_vocano(self, log_pval_thresh=5, log_fc_thresh=0.5, test="bimod", use_genes=None, use_raw=False):
    comparisons, ncells = self.compare_expression_between_groups(test=test, use_genes=use_genes, use_raw=use_raw)
    comparisons["pval"] = -np.log10(comparisons["pval"])
    ymax = comparisons["pval"].replace([np.inf, -np.inf], np.nan).dropna(how="all").max()
    n_clusts = len(np.unique(self._clusts))

    for i, c in enumerate(np.unique(self._clusts)):
        ax = plt.subplot(1, n_clusts, i+1)
        curr_vals = comparisons[comparisons["cluster"] == c]
        m = 6
        plt.plot(curr_vals["log_fc"], curr_vals["pval"], 'ko', markersize=m, markeredgewidth=0, linewidth=0)
        good_genes = curr_vals.loc[curr_vals["pval"] > log_pval_thresh, :]
        for g in good_genes.index:
            if g == "Egr2":
                print("Clu=%d,Name=%s,logFC=%f,Pval=%f" % (c, str(g), good_genes.loc[g, "log_fc"], good_genes.loc[g, "pval"]))
        for g in good_genes.index:
            x = good_genes.loc[g,"log_fc"]
            y = good_genes.loc[g,"pval"]
            plt.plot(x, y, 'go', markersize=m, markeredgewidth=0, linewidth=0)
        good_genes = good_genes.loc[good_genes["log_fc"].abs() > log_fc_thresh, :]

        for g in good_genes.index:
            x = good_genes.loc[g, "log_fc"]
            y = good_genes.loc[g, "pval"]
            plt.plot(x, y, 'ro', markersize=m, markeredgewidth=0, linewidth=0)
            plt.text(x, y, str(g), fontsize=18)
        plt.xlim([-2, 2])
        plt.ylim([-0.5, 1.2*ymax])
        ax.set_xticks([-2, 0, 2])
        if i > 0:
            ax.get_yaxis().set_visible(False)
        sns.despine()
        plt.tick_params(axis='both', which='major', labelsize=18)


# Plot dotplot for expression across clusters
def dotplot_expression_across_clusters(self, gene_names, scale_max=500, cmap=plt.cm.get_cmap('viridis'), clust_order=False):
    n_genes = len(gene_names)
    n_clusts = len(np.unique(self._clusts))
    uniq_clusts, clust_counts = np.unique(self._clusts, return_counts=True)

    avg = []  # averge expression value of the genes
    num = []  # number of cells epxressed these genes

    for i in range(n_genes):
        expr = self.get_expr_for_gene(gene_names[i], scaled=True).values
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        avg_expr = d.groupby("cluster").mean()
        avg_expr /= avg_expr.sum()
        avg_expr = avg_expr.values.flatten()
        num_expr = d.groupby("cluster").apply(lambda x: (x["expr"] > 0).sum()).values.astype(np.float)
        num_expr /= clust_counts
        avg.append(avg_expr)
        num.append(num_expr)
    avg = np.vstack(avg)
    num = np.vstack(num)

    if clust_order:
        pos = []
        for i in range(n_genes):
            idx = np.argsort(-avg[i,:])
            for k in idx:
                if k not in pos:
                    pos.append(k)
                    break
        print("Number of Genes: %d\nNumber of Clusters: %d" % avg.shape)
        print("Indexes of Cluster shown: ", pos)
        num = num[:, pos]
        avg = avg[:, pos]
        pos = range(num.shape[1])
    else:
        pos = range(n_clusts)

    for i in range(n_genes):
        plt.scatter(pos, -i*np.ones_like(pos), s=num[i, :]*scale_max, c=avg[i, :],
                    cmap=cmap, vmin=0, vmax=avg[i, :].max(), lw=0)

    plt.yticks(-np.array(range(n_genes)), gene_names)
    plt.axes().set_xticks(pos)
    plt.axes().set_xticklabels(pos)
    plt.xlabel('Cluster')


# Plot array of genes x clusters
def plot_expression_across_clusters(self, gene_names, plot_type="bar", figsize=None,
                                    clust_order=None, show_frame=True, palette='gray'):
    if clust_order is not None:
        clusts = self._clusts.copy()
        for i,c in enumerate(clust_order):
            clusts[self._clusts == c] = i
    else:
        clusts = self._clusts

    n_genes = len(gene_names)

    if figsize is None:
        f, ax = plt.subplots(n_genes, 1)
    else:
        f, ax = plt.subplots(n_genes, 1, figsize=figsize)
    f.tight_layout()

    for i in range(n_genes):
        expr = self.get_expr_for_gene(gene_names[i], scaled=False).values
        # plt.title(name)
        d = pd.DataFrame(np.array([expr, clusts]).T, columns=["expr", "cluster"])
        if plot_type == "bar":
            sns.barplot(x="cluster", y="expr", data=d, ax=ax[i], capsize=.2, errwidth=1, palette=palette)
        elif plot_type == "violin":
            sns.violinplot(x="cluster", y="expr", data=d, ax=ax[i], capsize=.2, errwidth=1)
        # get rid of the frame
        if not show_frame:
            for spine in ax[i].spines.values():
                spine.set_visible(False)
            ax[i].get_xaxis().set_visible(False)
            ax[i].get_yaxis().set_visible(True)
            ax[i].set_ylabel(gene_names[i])
            if i == n_genes-1:
                ax[i].tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')
            else:
                ax[i].tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')


# Plot bar chart for gene expression
def plot_bar_gene_expression(self, gene_names, nrow=None, ncol=None, ylim=None, figsize=(5,5), cmap=None):
    def _bar_plot(name, ylim=None, ax=None):
        expr = self.get_expr_for_gene(name, scaled=False).values
        # plt.title(name)
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        if cmap is None:
            sns.barplot(x="cluster", y="expr", data=d, ax=ax)
        else:
            sns.barplot(x="cluster", y="expr", data=d, ax=ax,palette=cmap)
        if ax is None:
            ax = plt.axes()
        ax.set_title(name)
        if ylim is not None:
            ax.set_ylim([-1, ylim])
        sns.despine(ax=ax)
        ax.set_xlabel("")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        f,ax = plt.subplots(int(nrow), int(ncol), figsize=figsize)
        f.tight_layout()
        ax = np.array(ax).flatten()
        for i, name in enumerate(gene_names):
            _bar_plot(name, ylim, ax=ax[i])
    else:
        _bar_plot(gene_names, ylim)


# Plot heatmap
def plot_heatmap(self, gene_names, cmap=plt.cm.get_cmap('bwr'), fontsize=16,
                 use_imshow=False, ax=None, show_vlines=True, annotation=None):
    if self._active_cells is not None:
        input_data = self._data.iloc[self._active_cells, :]
    else:
        input_data = self._data

    if ax is None:
        ax = plt.axes()

    clust_sizes = [sum(self._clusts == i) for i in np.unique(self._clusts)]
    data = np.vstack([input_data.iloc[self._clusts == i, :].loc[:, gene_names].values for i in np.unique(self._clusts)]).T

    if use_imshow:
        ax.imshow(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap, interpolation='none', aspect='auto')
    else:
        im = ax.pcolor(np.flipud(zscore(data, axis=1)), vmin=-2.5, vmax=2.5, cmap=cmap)
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
    # ax.figure.colorbar(im, ax=ax)
    # ax.get_xaxis().set_fontsize(fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    return im

# Plot violin plot for gene expression
def plot_violin_gene_expression(self, gene_names, nrow=None, ncol=None, ylim=None):
    def _violin_plot(name, ylim=None):
        expr = self.get_expr_for_gene(name, scaled=False).values
        plt.title(name)
        d = pd.DataFrame(np.array([expr, self._clusts]).T, columns=["expr", "cluster"])
        sns.violinplot(x="cluster", y="expr", data=d)
        if ylim is not None:
            plt.ylim([-1, ylim])

    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        for i, name in enumerate(gene_names):
            plt.subplot(nrow, ncol, i+1)
            _violin_plot(name, ylim)
    else:
        _violin_plot(gene_names, ylim)


# Plot the expression of a single gene in tSNE space
def plot_tsne_gene_expression(self, gene_names, scaled=True, nrow=None, ncol=None, s=10):
    if isinstance(gene_names, list):
        if nrow is None and ncol is None:
            nrow = np.ceil(np.sqrt(len(gene_names)))
            ncol = nrow
        for i,name in enumerate(gene_names):
            plt.subplot(nrow, ncol, i+1)
            expr = self.get_expr_for_gene(name, scaled=scaled)
            plt.title(name, fontsize=16)
            plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=expr, cmap=plt.cm.get_cmap('jet'), s=s, lw=0)
            plt.axis('off')
    else:
        expr = self.get_expr_for_gene(gene_names, scaled=scaled)
        plt.title(gene_names, fontsize=16)
        plt.scatter(self._tsne[:, 0], self._tsne[:, 1], c=expr, cmap=plt.cm.get_cmap('jet'), s=s, lw=0)
    # plt.axis('off')


# ==== CELL TYPING WITH MORPHOLOGY ====
# Plot heatmap of gene markers of each cluster
def plot_heatmap_with_labels(self, degenes, cmap=plt.cm.get_cmap('jet'), show_axis=True,
                             show_top_ticks=True, use_labels=None, font_size=15, annotation=None):
    g = plt.GridSpec(2, 1, wspace=0.01, hspace=0.01, height_ratios=[0.5, 10])
    cluster_array = np.expand_dims(np.sort(self._clusts), 1).T
    ax1 = plt.subplot(g[0])
    ax1.imshow(cluster_array, aspect='auto', interpolation='none', cmap=cmap)
    if show_top_ticks:
        locations = []
        for i in np.unique(self._clusts):
            locs = np.median(np.argwhere(cluster_array == i)[:, 1].flatten())
            locations.append(locs)
        ax1.xaxis.tick_top()
        if use_labels is not None:
            plt.xticks(locations, use_labels)
        else:
            plt.xticks(locations, np.unique(self._clusts))
        ax1.get_yaxis().set_visible(False)
    # ax.axis('off')

    ax2 = plt.subplot(g[1])
    im = plot_heatmap(self, list(degenes), fontsize=font_size, use_imshow=False, ax=ax2, annotation=annotation)
    plt.colorbar(im, ax=[ax1, ax2])
    if not show_axis:
        plt.axis('off')


# Plot 2D polygon figure indicating gene expression pattern
def plot_poly_cells_expression(nissl, hulls, expr, cmap, good_cells=None, width=2, height=9, figscale=10, alpha=1, vmin=0, vmax=None):
    # define figure dims
    plt.figure(figsize=(figscale*width/float(height), figscale))
    # get polygon from convexhull
    polys = [hull_to_polygon(h) for h in hulls]
    # filter based on cells
    if good_cells is not None:
        polys = [p for i, p in enumerate(polys) if i in good_cells]
    p = PatchCollection(polys, alpha=alpha, cmap=cmap, linewidths=0)
    p.set_array(expr)
    if vmax is None:
        vmax = expr.max()
    else:
        vmax = vmax
    p.set_clim(vmin=vmin, vmax=vmax)
    plt.gca().add_collection(p)
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.axis('off')


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster(nissl, hulls, colors, cmap, other_cmap='gray',
                            good_cells=None, width=2, height=9, figscale=10,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height),figscale))
    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
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
    # return polys


# Plot 2D polygon figure indicating cell typing and clustering
def plot_poly_cells_cluster_with_plaque(nissl, plaque, hulls, colors, cmap,
                            good_cells=None, width=2, height=9, figscale=10,
                            rescale_colors=False, alpha=1, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height),figscale))
    polys = [hull_to_polygon(h) for h in hulls]

    if good_cells is not None:
        others = [p for i, p in enumerate(polys) if i not in good_cells]
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
    plt.imshow(plaque.T, cmap=plt.cm.get_cmap('binary'))
    plt.imshow(nissl.T, cmap=plt.cm.get_cmap('gray_r'), alpha=0.15)
    plt.gca().add_collection(p)
    # plt.gca().add_collection(o)
    plt.axis('off')
    # return polys


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
    poly = Polygon(k*(np.array(pts) - cent) + cent, edgecolor='k', linewidth=1)
    # poly.set_capstyle('round')
    return poly


# Get Convexhull for each cell
def get_qhulls(labels):
    hulls = []
    coords = []
    centroids = []
    print('Geting ConvexHull...')
    for i, region in enumerate(regionprops(labels)):
        if 100000 > region.area > 1000:
            hulls.append(ConvexHull(region.coords))
            coords.append(region.coords)
            centroids.append(region.centroid)
    num_cells = len(hulls)
    print(f"Used {num_cells} / {i + 1}")
    return hulls, coords, centroids


# Plot cells of each cluster
def plot_cells_cluster(nissl, coords, good_cells, colors, cmap, width=2, height=9, figscale=100, vmin=None, vmax=None):
    plt.figure(figsize=(figscale*width/float(height), figscale))
    img = -1 * np.ones_like(nissl)
    curr_coords = [coords[k] for k in range(len(coords)) if k in good_cells]
    for i, c in enumerate(curr_coords):
        for k in c:
            if k[0] < img.shape[0] and k[1] < img.shape[1]:
                img[k[0], k[1]] = colors[i]
    plt.imshow(img.T, cmap=cmap, vmin=-1, vmax=colors.max())
    plt.axis('off')


# Plot cells of each cluster as dots
def plot_dot_cells_cluster_2d(centroids, good_cells, colors, cmap):
    good_centroids = []
    for i in range(len(centroids)):
        if i in good_cells:
            good_centroids.append(centroids[i])

    transformed_centroids = ut.rc2xy_2d(np.array(good_centroids))

    plt.figure(figsize=(20, 10))
    plt.scatter(transformed_centroids[:, 0], transformed_centroids[:, 1], s=100, c=colors, cmap=cmap)
    plt.axis('off')
    plt.show()


# Get colormap
def get_colormap(colors):
    pl = sns.color_palette(colors)
    cmap = ListedColormap(pl.as_hex())

    sns.palplot(sns.color_palette(pl))
    return cmap

