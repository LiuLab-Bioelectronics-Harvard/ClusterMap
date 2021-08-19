from .utils import *
from .preprocessing import *
from .stitch import *
from .cell_typing import *
from .tissue_mapping import *
import random
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.morphology import convex_hull_image
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import Delaunay
from sklearn.metrics import adjusted_rand_score

class ClusterMap():

    def __init__(self, spots, dapi, gene_list, num_dims):
        
        '''
        params :    - spots (dataframe) = columns should be 'spot_location_1', 'spot_location_2',
                     ('spot_location_3'), 'gene'
                    - dapi (ndarray) = original dapi image
                    - gene_list (1Darray) = list of gene identities (encoded as ints)
                    - num_dims (int) = number of dimensions for cell segmentation (2 or 3)
        '''

        # self.spots = pd.read_csv(spot_path)
        # self.dapi = tifffile.imread(dapi_path)
        
        # if len(self.dapi.shape) == 3:
        #     self.dapi = np.transpose(self.dapi, (1,2,0))
        self.spots = spots
        self.dapi = dapi
        self.dapi_binary, self.dapi_stacked = binarize_dapi(self.dapi)
        self.gene_list = gene_list
        self.num_dims = num_dims
    
    def preprocess(self):
        preprocessing_data(self.spots, self.dapi_binary)

    def segmentation(self, R, d_max, add_dapi=False):
        
        '''
        params :    - R (float) = rough radius of cells
                    - d_max (float) = maximum distance to use (often chosen as R)
                    - add_dapi (bool) = whether or not to add Dapi points for DPC
        '''
        
        spots_denoised = self.spots.loc[self.spots['is_noise']==0,:].copy()
        spots_denoised.reset_index(inplace=True)
        
        print('Computing NGC coordinates')
        ngc = NGC(spots_denoised, R, self.num_dims, self.gene_list)
        if add_dapi:
            print('Adding DAPI points')
            all_coord, all_ngc = add_dapi_points(self.dapi_binary, spots_denoised, ngc, self.num_dims)
            print('DPC')
            cell_ids = DPC(all_coord, all_ngc, R, d_max)
        else:
            spatial = np.array(spots_denoised[['spot_location_1', 'spot_location_2', 'spot_location_3']]).astype(np.float32)
            print('DPC')
            cell_ids = DPC(spatial, ngc, R, d_max)
        self.spots['clustermap'] = -1

        # Let's keep only the spots' labels
        self.spots.loc[spots_denoised.loc[:, 'index'], 'clustermap'] = cell_ids[:len(ngc)]
    
    def plot_segmentation(self):
        spots_repr = self.spots.loc[self.spots['clustermap']>=0,:]
        plt.figure(figsize=(20,20))
        palette = sns.color_palette('gist_ncar', len(spots_repr['clustermap'].unique()))
        sns.scatterplot(x='spot_location_1', y='spot_location_2', data=spots_repr, hue='clustermap', palette=palette, legend=False)
        plt.title('Segmentation')
        plt.show()
    def save(self, path_save):
        self.spots.to_csv(path_save, index=False)
        

class StitchSpots():
    def __init__(self, path_res, path_config, res_name):

        '''
        params :    - path_res (str) = root path of the results of AutoSeg's segmentation
                    - path_config (str) = path of tile configuration
                    - res_name (str) = name of the column where AutoSeg's results are stored in each dataset
        '''
        
        self.path_res = path_res
        self.path_config = path_config
        self.res_name = res_name
       
    def gather_tiles(self):
        print('Gathering tiles')
        self.spots_gathered = gather_all_tiles(self.path_res, self.res_name)

    def stitch_tiles(self):
        print('Loading config')
        self.config = load_tile_config(self.path_config)
        print('Stitching tiles')
        self.img, self.num_col, self.num_row = create_img_label(self.config)
        self.spots_all = stitch_all_tiles(self.spots_gathered, self.img, self.num_col, self.num_row, self.config, self.res_name)
    
    def plot_stitched_data(self, figsize=(16,10), s=0.5):
        spots_all_repr = self.spots_all.loc[self.spots_all['cellid']>=0,:]
        plt.figure(figsize=figsize)
        palette = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for k in range(spots_all_repr['cellid'].unique().shape[0])]
        sns.scatterplot(x='spot_merged_1', y='spot_merged_2', data=spots_all_repr, hue='cellid', legend=False, s=s, palette=palette)
        plt.title('Stitched dataset')
        plt.show()
    
    def save_stitched_data(self, path_save):
        self.spots_all.to_csv(path_save, index=False)

class CellTyping():
    def __init__(self, spots_stitched_path, var_path, gene_list, method, use_z):
        
        '''
        Perform cell typing on the stitched dataset.

        params :    - spots_stitched_path (str) = path of the results
                    - gene_list (list of ints) = genes used
                    - method (str) = name of column of results
        '''
        
        self.spots_stitched_path = spots_stitched_path
        self.spots = pd.read_csv(spots_stitched_path)
        self.var = pd.read_csv(var_path, header=None)
        self.var = pd.DataFrame(index=self.var.iloc[:,0].to_list())
        self.gene_list = gene_list
        self.method = method
        self.use_z = use_z
        self.markers = None
        self.adata = None
        self.palette = None
        self.cell_shape = None
    
    def gene_profile(self, min_counts_cells=16, min_cells=10, plot=False):
        
        '''
        Generate gene profile and find cell centroids. Perform normalization.

        params :    - min_count_cells (int) = minimal number of counts of a cell to be not discarded
                    - min_cells (int) = filter genes and erase the ones that are expressed in less than min_cells cells. 
                    - plot (bool) = whether to plot the gene profile before clustering        
        '''
        
        print('Generating gene expression and finding cell centroids')
        gene_expr, obs = generate_gene_profile(self.spots, self.gene_list, use_z=self.use_z, method=self.method)
        print('Normalizing')
        adata = normalize_all(gene_expr, obs, self.var, min_counts_cells=min_counts_cells, min_cells=min_cells, plot=plot)
        self.adata = adata

    def cell_typing(self,n_neighbors=20, resol=1, n_clusters=None, type_clustering='leiden'):

        '''
        Performs cell typing.

        params :    - n_neighbors (20) = number of neighbors to use for scanpy pp.neighbors
                    - resol (float) = resolution of Leiden of Louvain clustering
                    - n_clusters (int) = number of clusters to determine (in case we are using agglomerative clustering)
                    - type_clustering (str) = type of clustering for cell typing. Can be 'leiden', 'louvain', or 'hierarchical'
        '''

        sc.tl.pca(self.adata)
        sc.pp.neighbors(self.adata, n_neighbors=n_neighbors, n_pcs=10, random_state=42)
        sc.tl.umap(self.adata, random_state=42)
        if type_clustering == 'leiden':
            print('Leiden clustering')
            sc.tl.leiden(self.adata, resolution=resol, random_state=42, key_added='cell_type')
        elif type_clustering == 'louvain':
            sc.tl.louvain(self.adata, resolution=resol, random_state=42, key_added='cell_type')
        else:
            agg = AgglomerativeClustering(n_clusters=n_clusters, 
                                         distance_threshold=None,
                                         affinity='euclidean').fit(self.adata.X)
            
            self.adata.obs['cell_type'] = agg.labels_.astype('category')

        
        cluster_pl = sns.color_palette("tab20_r", 15)
        self.palette = cluster_pl
        sc.pl.umap(self.adata, color='cell_type', legend_loc='on data',
                    legend_fontsize=12, legend_fontoutline=2, frameon=False, 
                    title=f'clustering of cells : {type_clustering}', palette=cluster_pl, save=False)
        sc.tl.rank_genes_groups(self.adata, 'cell_type', method='t-test')

        # Pick markers 
        markers = []
        temp = pd.DataFrame(self.adata.uns['rank_genes_groups']['names']).head(5)
        for i in range(temp.shape[1]):
            curr_col = temp.iloc[:, i].to_list()
            markers = markers + curr_col
            print(i, curr_col)
            
        self.markers = markers
       
    
    def plot_cell_typing_spots(self,save_path, figsize=(16,10), s=10):
        
        '''
        Plot the spots colored by their cell typing

        params :    - figsize (tuple) = size of the figure
                    - s (int) = width of each point
        '''

        cell_typing2spots(self.adata, self.spots, method=self.method)
        spots_repr = self.spots.loc[self.spots['cell_type']!=-1,:]
        plt.figure(figsize=(figsize))
        sns.scatterplot(x='spot_merged_1', y='spot_merged_2', data=spots_repr, hue='cell_type', s=s, palette=self.palette[:len(np.unique(spots_repr['cell_type']))], legend=True)
        plt.title('Cell Typing')
        plt.savefig(save_path)
        plt.show()



class TissueMapping():
    def __init__(self, path_spots, path_all_spots):
        self.path_spots = path_spots
        self.path_all_spots = path_all_spots
        self.spots = pd.read_csv(path_spots)
        self.all_spots = pd.read_csv(path_all_spots)
        try:
            a = self.spots['cell_type']
        except KeyError as err:
            print('You need to do the cell typing first. See CellTyping() class')
        
    def compute_ncc(self, radius):
        print('Computing cell neighborhoods')
        self.ncc = neighboring_cell_types(self.spots, radius)
    
    def identify_tissues(self, n_neighbors=30, resol=0.1, figsize=(30,20), s=0.5):
        adata = sc.AnnData(self.ncc)
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X', random_state=42)
        sc.tl.umap(adata, random_state=42)

        print('Leiden clustering')
        sc.tl.leiden(adata, resolution=resol, random_state=42, key_added='tissue')
        sc.pl.umap(adata, color='tissue', palette='gist_ncar')
        adata.obs['index'] = np.array(self.spots['index'].astype(int))
        self.adata = adata
        
        print('Assign tissue label to each spot')
        tissue2spot(adata, self.all_spots, 'cellid')

        repres = self.all_spots.loc[self.all_spots['tissue']>=0,:]
        plt.figure(figsize=figsize)
        sns.scatterplot(x='spot_merged_1', y='spot_merged_2', data=repres, hue='tissue', s=s, palette=sns.color_palette('Paired', len(repres['tissue'].unique())))
        plt.title('Tissue regions')
        plt.show()
    
    def save_tissues(self, save_path):
        self.spots.to_csv(save_path, index=False)

class CellNiches():
    def __init__(self, centroid_path, gene_expr_path):
        self.centroid_path = centroid_path
        self.centroids = pd.read_csv(centroid_path)
        try:
            self.gene_expr = np.load(gene_expr_path)
        except KeyError as err:
            print('You need to apply cell typing first')
        self.tri = Delaunay(self.centroids[['cell_center_1', 'cell_center_2', 'cell_center_3']]).simplices
    
    def compute_counts(self):
        cell_types = np.unique(self.centroids['cell_type'])
        mean_cell_types = []
        counts_per_cell_type = []
        for cell_type in cell_types:
            print(f'Processing cell type : {cell_type}')
            cells = np.unique(self.centroids.loc[self.centroids['cell_type']==cell_type,:].index.to_list())
            counts = np.zeros((len(cells),len(cell_types)))
            for i,cell in enumerate(cells):
                connected_idx = [cell in self.tri[i] for i in range(len(self.tri))]
                connected_cells = np.unique(self.tri[connected_idx])
                connected_types = self.centroids.loc[connected_cells,:].groupby('cell_type').size()
                counts[i, connected_types.index.to_list()] = np.array(connected_types)
            counts_per_cell_type.append(counts)
            mean = np.mean(counts, axis=0)
            mean_cell_types.append(mean)
        self.mean_cell_types = mean_cell_types
        self.counts_per_cell_type = counts_per_cell_type
    
    def plot_stats(self, figsize=(20,25), lw=2, size=1):
        fig, axes = plt.subplots(nrows=len(self.centroids['cell_type'].unique()), ncols=1, figsize=figsize)
        plt.rc('font', size=15)
        plt.rc('axes', labelsize=10)
        plt.rc('xtick', labelsize=10)

        for i in range(len(self.mean_cell_types)):
            dg = pd.DataFrame(zip(np.round(self.mean_cell_types[i],2), np.arange(len(self.mean_cell_types[0]))), columns=['value', 'cell_type'])
            pl = sns.barplot(x='cell_type', y='value', data=dg, ax=axes[i],fc='white', ec='black',lw=lw)
            pl.bar_label(pl.containers[0])
        
            df = pd.DataFrame(zip(np.round(np.ravel(self.counts_per_cell_type[i]),2), len(self.counts_per_cell_type[i])*list(np.arange(len(self.mean_cell_types[0])))), columns=['value', 'cell_type'])
            sns.stripplot(x='cell_type', y='value', data=df, palette='pastel',ax=axes[i], size=size)
            axes[i].set_frame_on(False)
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Cell Type '+str(i))
            axes[i].axhline()
        axes[-1].set_xlabel('Cell types')
        plt.xticks(np.arange(len(self.mean_cell_types)), ['Cell Type '+str(i) for i in range(len(self.mean_cell_types))])
        plt.show()
                
    def discover_subclusters(self, resol_gene, resol_niche, n_neighbors):
        '''
        Compare subcelltyping from (1) gene expression subclustering and (2) cell niche expression clustering

        params :    - resol_gene (float) = Leiden resolution for gene expression clustering
                    - resol_niche (float) = Leiden resolution for niche clustering
                    - n_neighbors (int) = number of neighbors to use to ompute Leiden's graph
        '''
        ari_list = []
        cell_types = np.unique(self.centroids['cell_type'])
        for ct in cell_types:
            print(f'Processing cell type : {ct}')
            ### Clustering with gene expression
            idx = np.array(self.centroids.loc[self.centroids['cell_type']==ct, :].index, dtype=int)
            expr_ct = self.gene_expr[idx]
            adata_g = sc.AnnData(expr_ct)
            sc.pp.neighbors(adata_g, n_neighbors=n_neighbors, random_state=42, use_rep='X')
            sc.tl.umap(adata_g, random_state=42)
            sc.tl.leiden(adata_g, resolution=resol_gene, random_state=42, key_added='subclusters')
            sc.pl.umap(adata_g, color='subclusters')
            
            ## Clustering with niche expressions
            adata_n = sc.AnnData(self.counts_per_cell_type[int(ct)])
            sc.pp.neighbors(adata_n, n_neighbors=30, random_state=42, use_rep='X')
            sc.tl.leiden(adata_n, resolution=resol_niche, random_state=42, key_added='niche_subclusters')
            adata_g.obs['niche_subclusters'] = adata_n.obs['niche_subclusters']
            sc.pl.umap(adata_g, color='niche_subclusters')
            
            ari = adjusted_rand_score(adata_g.obs['subclusters'], adata_n.obs['niche_subclusters'])
            print(f'Adjusted Rand index for cell type {ct} = {ari}')
            
            ari_list.append(ari)