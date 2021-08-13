import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.morphology import square,erosion,reconstruction
from itertools import product
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

def binarize_dapi(dapi):
    '''
    Binarize raw dapi image

    params : - dapi (ndarray) = raw DAPI image

    returns : - dapi_binary (ndarray) = binarization of Dapi image
              - dapi_stacked (ndarray) =  2D stacked binarized image
    '''
    degree = len(dapi.shape)
    if degree==2:
        #binarize dapi
        dapi_marker=erosion(dapi, square(5))
        dapi_recon=reconstruction(dapi_marker,dapi)
        thresh = threshold_otsu(dapi_recon)
        binary = dapi_recon >= thresh
        dapi_binary=np.array(binary).astype(float)
        dapi_stacked = dapi_binary
    else:
        dapi_binary=[]
        for t in np.arange(dapi.shape[2]):
            dapi_one_page=dapi[:,:,t]
            dapi_marker=erosion(dapi_one_page, square(5))
            dapi_recon=reconstruction(dapi_marker,dapi_one_page)
            thresh = threshold_otsu(dapi_recon)
            binary = dapi_recon >= thresh
            dapi_binary.append(binary) #z,y,x
            ### erosion on dapi binary
        dapi_binary = np.array(dapi_binary).transpose((1,2,0))#y,x,z        
        dapi_stacked = np.amax(dapi_binary, axis=2)
    return(dapi_binary, dapi_stacked)

def preprocessing_data(spots, dapi_binary):
    '''
    Apply preprocessing on spots, thanks to dapi. 
    We remove the 10% spots with lowest density

    params :    - spots (dataframe) = spatial locations and gene identity
                - dapi_binary (ndarray) = binarized dapi image
    
    returns :   - spots (dataframe)
    '''
    sampling_mat = np.zeros(dapi_binary.shape)
    if len(dapi_binary.shape)==3:
        for ii,jj,kk in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1]),range(sampling_mat.shape[2])):
            if ii%3==1 and jj%3==1 and kk%3==1:
                sampling_mat[ii,jj,kk] = 1
        dapi_sampled = dapi_binary*sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        
        all_points = np.concatenate((np.array(spots.loc[:, ['spot_location_2', 'spot_location_1','spot_location_3']]), dapi_coord), axis=0)
        
        #compute neighbors within radius for local density
        knn = NearestNeighbors(radius=20)
        knn.fit(all_points)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1','spot_location_3']])
        neigh_dist, neigh_array = knn.radius_neighbors(spots_array) 
        
        #global low-density removal
        dis_neighbors=[ii.sum(0) for ii in neigh_dist]
        thresh = np.percentile(dis_neighbors, 10)
        noisy_points = np.argwhere(dis_neighbors<=thresh)[:,0]
        spots['is_noise'] = 0
        spots.loc[noisy_points, 'is_noise'] = -1
        
        #LOF
        res_num_neighbors = [i.shape[0] for i in neigh_array]
        thresh = np.percentile(res_num_neighbors, 10)
        clf = LocalOutlierFactor(n_neighbors=int(thresh),contamination=0.1)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1','spot_location_3']])
        y_pred = clf.fit_predict(spots_array)
        spots.loc[y_pred==-1,'is_noise']=-1
        
        #spots in DAPI as inliers
        test=dapi_binary[list(spots_array[:,0]-1),list(spots_array[:,1]-1),list(spots_array[:,2]-1)]
        spots.loc[test==True,'is_noise']=0
    else:
        for ii,jj in product(range(sampling_mat.shape[0]), range(sampling_mat.shape[1])):
            if ii%3==1 and jj%3==1:
                sampling_mat[ii,jj] = 1
    
        dapi_sampled = dapi_binary*sampling_mat
        dapi_coord = np.argwhere(dapi_sampled > 0)
        
        all_points = np.concatenate((np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']]), dapi_coord), axis=0)
        
        #compute neighbors within radius for local density
        knn = NearestNeighbors(radius=20)
        knn.fit(all_points)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']])
        neigh_dist, neigh_array = knn.radius_neighbors(spots_array) 
        
        #global low-density removal
        dis_neighbors=[ii.sum(0) for ii in neigh_dist]
        res_num_neighbors = [ii.shape[0] for ii in neigh_array]

        thresh = np.percentile(dis_neighbors, 10)
        noisy_points = np.argwhere(dis_neighbors<=thresh)[:,0]
        spots['is_noise'] = 0
        spots.loc[noisy_points, 'is_noise'] = -1
        
        #LOF
        thresh = np.percentile(res_num_neighbors, 10)
        clf = LocalOutlierFactor(n_neighbors=int(thresh),contamination=0.1)
        spots_array = np.array(spots.loc[:, ['spot_location_2', 'spot_location_1']])
        y_pred = clf.fit_predict(spots_array)
        spots.loc[y_pred==-1,'is_noise']=-1
        
        #spots in DAPI as inliers
        test=dapi_binary[list(spots_array[:,0]-1),list(spots_array[:,1]-1)]
        spots.loc[test==True,'is_noise']=0

    return(spots)