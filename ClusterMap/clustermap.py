from .utils import *
from .preprocessing import *
import tifffile
import matplotlib.pyplot as plt
import seaborn as sns

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


