"""
A python package of STARMap bioinformatics analysis pipeline.

"""

# Import packages
import os
import numpy as np
import pandas as pd
import textwrap as tw
from anndata import AnnData
# from .viz import *
# from .analyze import *


class STARMapDataset(object):
    """This is the fundamental class"""

    def __init__(self):
        self._raw_data = None  # raw data
        self._data = None  # data that has been normalized (log + total count)
        self._scaled = None  # scaled data
        self._pca = None  # pca (object) for cells
        self._transformed_pca = None  # pca (values) for cells
        self._tsne = None  # tsne for cells
        self._clusts = None  # per cell clustering
        self._meta = None  # per cell metadata
        self._meta_out = None  # output metadata
        self._nexpt = 0  # number of experiment

        self._nissl = None  # nissl channel image
        self._hulls = None  # convexhull generated from the label image (nissl or others)

        self._all_ident = None  # original identity of all cells (including those filtered out)

        # binary array of which cells/genes are included and which are filtered out
        self._good_cells = None
        self._good_genes = None
        self._active_cells = None
        self._ncell, self._ngene = 0, 0

    # ==== INPUT FUNCTIONS ====
    # Loading data into the object
    def add_data(self, data, group=None, tpm_norm=False, use_genes=None, cluster_id_path=None):
        """
        Add data to data matrix, keeping track of source.
        Inputs:
            group: numeric id of experiment
            tpm_norm: normalize raw data to TPM (transcripts per million)
            use_genes: only include listed genes
            cluster_id_path: load cluster identities from CSV file (of format cell_num,cluster_id,cluster_name per line)
        """
        reads_per_cell = data.sum(axis=1)  # row sum
        genes_per_cell = (data > 0).sum(axis=1)  # genes with reads per cell (row sum)
        # cells_per_gene = (data > 0).sum(axis=0)  # cells with reads per gene (column sum)

        # transfer to tpm
        if tpm_norm:
            data = data / (data.sum().sum() / 1e6)  # can get how many reads per million

        # get expression profile for subset of genes
        if use_genes:
            data = data[use_genes]

        # get stats
        n_cells, n_genes = data.shape
        # construct metadata table with group field or not
        if group is not None:
            meta = pd.DataFrame(np.vstack((np.ones(n_cells, dtype=np.int) * self._nexpt,
                                           reads_per_cell, genes_per_cell, np.repeat(group, n_cells))).T,
                                columns=["orig_ident", "reads_per_cell", "genes_per_cell", "group"])
        else:
            meta = pd.DataFrame(np.vstack((np.ones(n_cells, dtype=np.int) * self._nexpt,
                                           reads_per_cell, genes_per_cell)).T,
                                columns=["orig_ident", "reads_per_cell", "genes_per_cell"])

        # load cluster information from files
        if cluster_id_path is not None:
            labels = pd.read_csv(cluster_id_path)
            meta['cluster_id'] = labels["Cluster_ID"]
            meta['cluster_name'] = labels['Cluster_Name']

        # assign dataframes to the analysis object
        if self._nexpt == 0:
            self._raw_data = data
            self._meta = meta
        else:  # add data to existing dataframe
            self._raw_data = self._data.append(data)
            self._meta = self._meta.append(meta)

        self._data = self._raw_data
        self._nexpt += 1
        self._ncell, self._ngene = self._data.shape

        self._all_ident = np.array(self._meta['orig_ident'])

        self._good_cells = np.ones((self._ncell,), dtype=np.bool)
        self._good_genes = np.ones((self._ngene,), dtype=np.bool)

        # add filtration field
        # self._meta['Keep'] = False

        # add meta_out
        self._meta_out = self._meta.copy()

    # Load meta data
    def add_meta_data(self, input_meta=None):
        if isinstance(input_meta, str):
            self._meta = pd.read_csv(input_meta)
        elif isinstance(input_meta, pd.DataFrame):
            self._meta = input_meta
        else:
            print("Please provide a valid path of cluster label file or a pandas dataframe!")

    # Add meta data field
    def add_meta_data_field(self, field_name, field_data):
        self._meta[field_name] = field_data
        self._meta.head()

    # Load cluster labels for cells
    def add_cluster_data(self, input_cluster=None):
        if isinstance(input_cluster, str):
            labels = pd.read_csv(input_cluster)
            self._meta['cluster_id'] = labels["Cluster_ID"]
            self._meta['cluster_name'] = labels['Cluster_Name']
            self._clusts = self._meta['cluster_id'].to_numpy(dtype=np.int32)
        elif isinstance(input_cluster, pd.DataFrame):
            self._meta['cluster_id'] = input_cluster["Cluster_ID"]
            self._meta['cluster_name'] = input_cluster['Cluster_Name']
            self._clusts = self._meta['cluster_id'].to_numpy(dtype=np.int32)
        else:
            print("Please provide a valid path of cluster label file or a pandas data frame!")

    # Map numeric cluster labels to user defined labels
    def map_cluster_data(self, input_dict, id_field="cluster_id", label_field="cluster_label"):
        if id_field not in self._meta.columns:
            self._meta[id_field] = self._clusts

        self._meta[label_field] = self._meta[id_field]
        self._meta = self._meta.replace({label_field: input_dict})

    # Add cluster labels to output meta table
    def add_cluster_label(self, input_dict, label_field):
        # obj meta table
        if label_field not in self._meta.columns:
            self._meta[label_field] = "NA"

        if label_field not in self._meta_out.columns:
            self._meta_out[label_field] = "NA"

        if self._active_cells is not None:
            self._meta[label_field].iloc[self._active_cells] = self._clusts
            replace_cells = np.argwhere(self._good_cells == True).flatten()
            self._meta_out[label_field].iloc[replace_cells[self._active_cells]] = self._clusts
        else:
            self._meta[label_field] = self._clusts
            self._meta_out.loc[self._good_cells, label_field] = self._clusts


        self._meta = self._meta.replace({label_field: input_dict})
        self._meta_out = self._meta_out.replace({label_field: input_dict})
        # fill NA
        is_na = self._meta_out[label_field].isnull()
        self._meta_out.loc[is_na, label_field] = 'NA'

    # Add locations
    def add_location(self, input_location):
        dims = ['r', 'c', 'z']

        for c in range(input_location.shape[1]):
            self._meta_out[dims[c]] = input_location[:, c]

    # ==== DATA ACCESS ====
    # Get genes
    def get_gene_names(self):
        return self._raw_data.columns

    # Get features in metadata
    def get_metadata_names(self):
        return self._meta.columns

    # Get specific feature in metadata
    def get_metadata(self, feature):
        return self._meta[feature]

    # Get metadata of specific cluster
    def get_metadata_by_cluster(self, clust_id):
        cells = self._clusts == clust_id
        return self._meta.iloc[cells, :]

    def get_metaout_by_experiment(self, expt_id):
        return self._meta_out.loc[self._meta_out["orig_ident"] == expt_id, :]

    # Get expression profile for specific gene
    def get_expr_for_gene(self, gene_name, scaled=True):
        if scaled:
            return np.log1p(self._raw_data[gene_name])
        else:
            return self._raw_data[gene_name]

    # Get expression profile for specific cell
    def get_expr_for_cell(self, cell_index, scaled=True):
        if scaled:
            return np.log1p(self._raw_data.iloc[cell_index, :])
        else:
            return self._raw_data.iloc[cell_index, :]

    # Get mean expression profile for all clusters
    def get_mean_expr_across_clusters(self, scaled=True):
        expr = [self.get_mean_expr_for_cluster(i, scaled) for i in range(max(self._clusts))]
        return pd.concat(expr, axis=1).transpose()

    # Get mean expression profile for specific cluster
    def get_mean_expr_for_cluster(self, clust_id, scaled=True):
        data = self.get_cells_by_cluster(clust_id, use_raw=True)
        if scaled:
            return np.log1p(data)
        else:
            return data

    # Get cell index for specific experiment
    def get_cells_by_experiment(self, idx, use_genes=None, use_scaled=False):
        condition = self._meta["orig_ident"] == idx
        expt_idx = np.argwhere(condition.to_numpy()).flatten()

        if use_scaled:
            data = self._raw_data
        else:
            data = self._data

        if use_genes:
            return data.iloc[expt_idx, :].loc[:, use_genes]
        else:
            return data.iloc[expt_idx, :]

    # Get expression profile of cells in specific cluster
    def get_cells_by_cluster(self, clust_id, use_raw=False):
        cells = self._clusts == clust_id
        if use_raw:
            return self._raw_data.iloc[cells, :]
        else:
            return self._data.iloc[cells, :]

    # Get cluster information of specific experiment
    def get_cluster_for_experiment(self, expt_id):
        return self._clusts[self._meta["orig_ident"] == expt_id]

    # Get cluster information base on meta
    def get_cluster_for_group(self, meta_value, meta_field="group"):
        return self._clusts[self._meta[meta_field] == meta_value]

    # Get cluster labels
    def get_cluster_labels(self, cell_type_names=None):
        if cell_type_names is None:
            cluster_labels = np.array([cell_type_names[i] for i in self._clusts])
        else:
            cluster_labels = np.array(['NA' for _ in self._clusts])
        return cluster_labels

    # Get cell indexs and cluster labels of specific experiment
    def get_cells_and_clusts_for_experiment(self, expt_id, colorby=None):
        if self._active_cells is not None:
            meta = self._meta.iloc[self._active_cells, :]
        else:
            meta = self._meta

        if colorby is not None:
            good_cells = meta.index[(meta["orig_ident"] == expt_id)].values
            colors = self._meta[colorby].loc[self._meta['orig_ident'] == expt_id].cat.codes
        else:
            good_cells = meta.index[(meta["orig_ident"] == expt_id)].values
            colors = self._clusts[meta["orig_ident"] == expt_id]
        return good_cells, colors

    def show_stat(self):
        no_gene = np.sum(self._meta["genes_per_cell"] == 0)
        no_read = np.sum(self._meta["reads_per_cell"] == 0)
        count_dict = {}
        for i in range(self._nexpt):
            count_dict[i] = sum(self._meta['orig_ident'] == i)

        message = tw.dedent(f"""\
        Number of cells: {self._ncell}
        Cells without gene: {no_gene}
        Cells without reads: {no_read}
        Number of genes: {self._ngene}
        Number of experiments: {self._nexpt}
        Cells in each experiment: {count_dict}
        """)
        print(message)

    def show_reads_quantile(self):
        read_quantile = self._meta['reads_per_cell'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        print(f"Reads per cell quantile:\n{read_quantile}")

    # ==== SUBSET FUNCTIONS ====
    # Get subset of data by using cell index
    def subset_by_cell(self, cell_ids):
        subset = STARMapDataset()
        subset._raw_data = self._raw_data.iloc[cell_ids, :]  # raw data
        subset._data = self._data.iloc[cell_ids, :]  # data that has been normalized (log + total count)
        subset._scaled = self._scaled.iloc[cell_ids, :]  # scaled data
        subset._meta = self._meta.iloc[cell_ids]  # per cell metadata
        subset._nexpt = self._nexpt
        subset._ncell, subset._ngene = subset._data.shape
        return subset

    # Get subset of data by using cluster id
    def subset_by_cluster(self, cluster_id):
        cell_ids = np.argwhere(np.array(self._clusts) == cluster_id).flatten()
        subset = self.subset_by_cell(cell_ids)
        return subset

    # ==== MANIPULATE CLUSTER ====
    # Merge each cluster into first in list
    def merge_multiple_clusters(self, clusts):
        for idx, c in enumerate(clusts[1:]):
            self._clusts[self._clusts == c] = clusts[0]
        temp = self._clusts.copy()
        # relabel clusters to be contiguous
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx
        self._clusts = temp

    # Merge cluster1 into cluster0
    def merge_clusters(self, clust0, clust1):
        self._clusts[self._clusts == clust1] = clust0
        temp = self._clusts.copy()
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx
        self._clusts = temp

    # Sort clusters with input order
    def order_clusters(self, orders):
        temp = self._clusts.copy()
        n_clusters = len(np.unique(self._clusts))
        for idx, c in enumerate(np.unique(self._clusts)):
            temp[self._clusts == c] = idx + n_clusters

        for idx, c in enumerate(np.unique(temp)):
            temp[temp == c] = orders[idx]
            print(f"{idx} --> {orders[idx]}")
        self._clusts = temp

    # ==== Data Structure ====
    # Transfer to AnnData object
    def transfer_to_anndata(self):

        # Get X (expression profile)
        if self._scaled is None:
            X = self._data.values
        else:
            X = self._scaled.values

        # Get obs (metadata of obervations)
        obs = self._meta
        # if self._meta_out is None:
        #     obs = self._meta
        # else:
        #     obs = self._meta_out

        adata = AnnData(X=X, obs=obs)
        if self._transformed_pca is not None:
            adata.obsm['X_pca'] = self._transformed_pca
        if self._tsne is not None:
            adata.obsm['X_umap'] = self._tsne
        if self._clusts is not None:
            adata.obs['clusts'] = self._clusts

        return adata


# ====Additional IO function====
# Load gene expression table (output files from sequencing pipeline)
def load_data(data_dir, prefix="Cell"):
    expr = pd.read_csv(os.path.join(data_dir, "cell_barcode_count.csv"), header=None)
    gene_names = pd.read_csv(os.path.join(data_dir, "cell_barcode_names.csv"), header=None)
    row_names = ["%s_%05d" % (prefix, i) for i in range(expr.shape[0])]
    names = gene_names[2]
    names.name = "Gene"
    return pd.DataFrame(data=expr.values, columns=names, index=row_names)


# Load gene expression table (output file from MATLAB pipeline)
def load_new_data(data_dir):
    expr = pd.read_csv(os.path.join(data_dir, "geneByCell.csv"), header=0, index_col=0)
    return expr.T


# Load gene expression table (clean)
def load_data_test(data_dir):
    print('temp')
