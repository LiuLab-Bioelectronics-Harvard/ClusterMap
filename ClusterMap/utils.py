import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from itertools import product

def NGC(spots, radius, num_dim, gene_list):
    if num_dim == 3:
        X_data = spots[['spot_location_1', 'spot_location_2', 'spot_location_3']]
    else:
        X_data = spots[['spot_location_1', 'spot_location_2']]
    knn = NearestNeighbors(radius=radius)
    knn.fit(X_data)
    spot_number = spots.shape[0]
    res_neighbors = knn.radius_neighbors(X_data, return_distance=False)

    res_ngc = np.zeros((spot_number, len(gene_list)))
    for i in range(spot_number):
        neighbors_i = res_neighbors[i]
        genes_neighbors_i = spots.loc[neighbors_i, :].groupby('gene').size()
        res_ngc[i, genes_neighbors_i.index.to_numpy() - np.min(gene_list)] = np.array(genes_neighbors_i)
        res_ngc[i] /= len(neighbors_i)
    return(res_ngc)

def add_dapi_points(dapi_binary, spots_denoised, ngc, num_dims):
    
    '''
    Add sampled points for Binarized DAPI image to improve local connectivities

    params :    - dapi_binary (ndarray) = Binarized DAPI image
                - spots_denoised (dataframe) = denoised dataset
                - nodes (list of ints) = nodes of the StellarGraph
                - node_emb (ndarray) = node embeddings
    returns :   - adata_g (AnnData) = anndata object with the new input for Leiden

    '''

    ### Sample dapi points
    sampling_mat = np.zeros(dapi_binary.shape)
    if num_dims==3:
        for ii,jj,kk in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1]),range(sampling_mat.shape[2])):
            if ii%5==2 and jj%5==2 and kk%5==2:
                sampling_mat[ii,jj,kk] = 1
        dapi_sampled = dapi_binary*sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        spots_points = spots_denoised.loc[:, ['spot_location_2', 'spot_location_1', 'spot_location_3']]
    else:
        for ii,jj in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1])):
            if ii%5==2 and jj%5==2:
                sampling_mat[ii,jj] = 1
        dapi_sampled = dapi_binary*sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        spots_points = spots_denoised.loc[:, ['spot_location_2', 'spot_location_1']]


    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(spots_points)
    neigh_ind = knn.kneighbors(dapi_coord, 1, return_distance=False)
    
    ### Create dapi embedding thanks to the embedding of the nearest neighbor
    dapi_ngc = ngc[neigh_ind[:,0]]
    
    ### Concatenate dapi embedding + <x,y,z> with spots embedding + <x,y,z>
    all_ngc = np.concatenate((ngc, dapi_ngc), axis=0)
    all_coord = np.concatenate((spots_points, dapi_coord), axis=0)
    return(all_coord, all_ngc)

def spearman_metric(x,y):
    return(spearmanr(x,y).correlation)

def DPC(spatial, ngc,  R, d_max, spearman_metric=spearman_metric):    
    # Compute spatial distance
    spatial_dist = np.array(cdist(spatial, spatial, metric='euclidean'), dtype=np.float32)
    
    # Compute genetic distance
    distances = 1/np.array(cdist(ngc, ngc, metric=spearman_metric), dtype=np.float32)
    distances *= spatial_dist

    # Compute densities rho
    densities = np.sum(np.maximum(distances - d_max, 0)*np.exp((-(distances/R)**2), dtype=np.float32), axis=1, dtype=np.float32)
    
    index_highest_density = np.argmax(densities)
    
    # Compute distance delta (initialize gamma to delta for computation efficiency)
    gamma = np.array([np.min(distances[i,densities>densities[i]]) if i!=index_highest_density else 0 for i in range(len(densities))], dtype=np.float32)

    # Update gamma
    gamma[index_highest_density] = distances[index_highest_density,np.argmax(gamma)]
    gamma *= densities

    # Identify cell centers
    gamma_sorted = np.sort(gamma)
    gamma_sorted = gamma_sorted[::-1]
    spots_sorted_by_gamma = np.argsort(gamma)
    spots_sorted_by_gamma = spots_sorted_by_gamma[::-1]

    # Find elbow
    sample_gamma_50 = gamma_sorted[::50]
    diffs = np.abs(np.diff(sample_gamma_50))
    ind_thresh = np.argwhere(diffs<100)[0][0]

    # Find the cell centers
    cell_centers_index = spots_sorted_by_gamma[:50*ind_thresh]

    # Assign the rest of the spots by descending density rho
    ind_densities_sorted = np.argsort(densities)
    ind_densities_sorted = ind_densities_sorted[::-1]
    ind_densities_sorted_remaining_points = ind_densities_sorted[np.setdiff1d(np.arange(len(densities)),cell_centers_index )]

    ind_spots_assigned = list(np.array(cell_centers_index, copy=True))
    cell_ids = np.zeros(len(densities))
    cell_ids[cell_centers_index] = np.arange(1,len(cell_centers_index)+1) 
    for ind_point in ind_densities_sorted_remaining_points:
        # Assign the cell label of the nearest spatial neighbor
        nearest_point = np.argmin(spatial_dist[ind_point, ind_spots_assigned])
        cell_ids[ind_point] = cell_ids[ind_spots_assigned[nearest_point]]
        ind_spots_assigned.append(ind_point)

    return(cell_ids)