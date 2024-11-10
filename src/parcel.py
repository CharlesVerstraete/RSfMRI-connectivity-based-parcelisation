# python 3.9.6
# -*- coding: utf-8 -*-
"""
parcel.py
=========

Description:
    Class object for RS-fMRI cannectivity based parcellation of brain regions.

Dependencies:
    - nilearn 0.10.4
    - numpy 1.23.5
    - matplotlib 3.6.2
    
Usage:
    $ python parcel.py [options]

Author:
    Your Name <your.email@domain.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""

# Import libraries

''' math package '''
import numpy as np
from scipy.spatial import distance
from scipy.sparse import csgraph
from numpy import linalg as LA
from joblib import Parallel, delayed

''' nifti package '''   
import nibabel as nib                                          
import nilearn.masking as nimsk
from nilearn.image import resample_img
from nilearn.input_data import NiftiLabelsMasker

''' clustering package '''
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score

''' stat package '''   
import pandas as pd
from scipy import stats
import statsmodels.stats.multicomp as mc 

''' visualisation package '''   
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from nilearn import plotting

''' file management package ''' 
import sys
import os 

import matplotlib
matplotlib.use('Qt5Agg')


class Parcel:
    def __init__(self, 
                 anat_src : str, 
                 func_src : str, 
                 roi_src : str, 
                 seg_src : str, 
                 subject : int,
                 zmap_computed : bool  = False,
                 output_dir : str = None,
                 ) -> None:
        ''' Initialize the Parcel object '''
        self.subject = subject

        self.anat_img = nib.load(anat_src)
        self.func_img = nib.load(func_src)
        self.roi_img = nib.load(roi_src)
        self.seg_img = nib.load(seg_src)

        self.anat_data = self.anat_img.get_fdata()
        self.func_data = self.func_img.get_fdata()
        self.anat_shape = self.anat_data.shape
        self.anat_affine = self.anat_img.affine
        self.func_shape = self.func_data.shape
        self.func_affine = self.func_img.affine

        self.roi_data = self.roi_img.get_fdata().astype(int)
        self.seg_data = self.seg_img.get_fdata().astype(int)

        # ROI mask processing
        self.roi_resampled = resample_img(
            self.roi_img, 
            target_affine=self.func_affine, 
            target_shape=self.func_shape[:3], 
            interpolation='nearest')
        self.roi_data_resampled = self.roi_resampled.get_fdata()
        self.roi_data_resampled = (self.roi_data_resampled > 0).astype(np.uint8)
        self.roi_resampled = nib.Nifti1Image(
            self.roi_data_resampled, 
            self.func_affine, 
            dtype=np.uint8
        )

        # Segmentation mask processing
        self.seg_resampled = resample_img(
            self.seg_img, 
            target_affine=self.func_affine, 
            target_shape=self.func_shape[:3],
            interpolation='nearest')
        self.seg_data_resampled = self.seg_resampled.get_fdata()
        self.seg_data_resampled = (self.seg_data_resampled - self.roi_data_resampled > 0).astype(np.uint8)
        self.seg_resampled = nib.Nifti1Image(
            self.seg_data_resampled, 
            self.func_affine,
            dtype=np.uint8
        )

        self.roi_time_series = nimsk.apply_mask(self.func_img, self.roi_resampled).T
        self.seg_time_series = nimsk.apply_mask(self.func_img, self.seg_resampled).T

        self.roi_nvoxels = self.roi_time_series.shape[1]
        self.roivox_distance = np.zeros((self.roi_nvoxels, self.roi_nvoxels))
        self.n_clusters = 0
        self.labels = []
        self.centroids = []
        self.silhouette_score = 0
        self.n_out = 0
        self.position_out = []


    def rm_nan(self, data : np.ndarray) -> np.ndarray:
        ''' Remove NaN values from the data and store position'''

        index_x = np.isnan(data).all(axis = 0)
        data_xnanless = np.delete(data, index_x, axis=1)
        index_y = np.isnan(data_xnanless).all(axis = 1)
        data_xynanless = np.delete(data_xnanless, index_y, axis=0)
        self._nOut = sum(index_y)
        self._posOut = np.where(index_y)[0]
        
        return data_xynanless

    def get_zmap(self, dir_path = None, n_jobs = -1) -> np.ndarray:
        ''' Calculate the z-map of the functional data '''
        def chunk_correlations(start_idx, end_idx):
            chunk = distance.cdist(
                self.roi_time_series[start_idx:end_idx], 
                self.seg_time_series, 
                metric="correlation"
            )
            return chunk
        # Calculate chunk size
        n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
        chunk_size = len(self.roi_time_series) // n_chunks

        # Parallel correlation computation
        chunks = Parallel(n_jobs=n_jobs)(
            delayed(chunk_correlations)(i, i + chunk_size)
            for i in range(0, len(self.roi_time_series), chunk_size)
        )

        correlation_map = np.vstack(chunks)
        correlation_map_nanless = self.rm_nan(correlation_map)
        zmap = np.arctanh(1 - correlation_map_nanless)
        # Compute ROI voxel distances in parallel
        roivox_distance = Parallel(n_jobs=n_jobs)(
            delayed(distance.cdist)(zmap[i:i+chunk_size], zmap, metric="correlation")
            for i in range(0, len(zmap), chunk_size)
        )
        self.roivox_distance = np.vstack(roivox_distance)
        if dir_path:
            np.save(os.path.join(dir_path, f'sub{self.subject:03d}-zmap.npy'), zmap)
            np.save("roivox_distance.npy", self.roivox_distance)

    def spectral_clustering(self, n_clusters : int, dir_path = None) -> None:
        ''' Perform spectral clustering on the z-map '''
        self.n_clusters = n_clusters
        spectral = SpectralClustering(n_clusters= self.n_clusters, affinity="precomputed_nearest_neighbors", n_jobs = -1, random_state = 6)
        self.labels = spectral.fit_predict(self.roivox_distance)
        self.averaged_timecourses = np.array([np.mean(self.roi_time_series[self.labels == i], axis = 0) for i in range(n_clusters)])
        self.silhouette_score = silhouette_score(self.roivox_distance, self.labels, metric = "correlation")
        if dir_path:
            np.save(os.path.join(dir_path, f'sub{self.subject:03d}-{self.n_clusters}clusters-labels.npy'), self.labels)
            np.save(os.path.join(dir_path, f'sub{self.subject:03d}-{self.n_clusters}clusters-averaged_timecourses.npy'), self.averaged_timecourses)
    
    def search_optimal_clusters(self, max_clusters : int = 10, dir_path = None) -> None:
        ''' Search for the optimal number of clusters '''
        silhouette_scores = []
        for n_clusters in range(2, max_clusters + 1):
            self.spectral_clustering(n_clusters)
            silhouette_scores.append(self.silhouette_score)
        df = pd.DataFrame({"n_clusters": range(2, max_clusters + 1), "silhouette_score": silhouette_scores})
        if dir_path:
            df.to_csv(os.path.join(dir_path, f'sub{self.subject:03d}-silhouette_scores.csv'))
    
    def get_nifti_labels(self, dir_path = None) -> None:
        ''' Get the nifti labels '''
        self.labels += 1
        if self.n_out != 0 :
            self.labels = np.insert(self.labels, self._posOut, -1)
        roi_parceled = nimsk.unmask(self.labels, self.roi_resampled)
        roi_parceled_data = roi_parceled.get_fdata()
        for i in range(self.n_clusters):
            roi_parceled_data_copy = roi_parceled_data.copy()
            roi_parceled_data_copy[roi_parceled_data_copy != i + 1] = 0
            subroi_parcel = nib.Nifti1Image(roi_parceled_data_copy, self.func_affine, dtype=np.uint8)
            if dir_path:
                nib.save(subroi_parcel, os.path.join(dir_path, f'sub{self.subject:03d}-parcel-{i+1}.nii'))
        if dir_path:
            nib.save(roi_parceled, os.path.join(dir_path, f'sub{self.subject:03d}-parcel.nii'))


    def compute_connectivity(self, sub_roi, dir_path=None, n_jobs=-1) -> None:
        """Compute connectivity matrix in parallel."""
        
        def process_chunk(chunk_start, chunk_size):
            chunk_results = np.zeros(chunk_size)
            for i in range(chunk_size):
                idx = chunk_start + i
                if idx < self.seg_time_series.shape[0]:
                    chunk_results[i] = np.arctanh(
                        np.corrcoef(
                            self.averaged_timecourses[sub_roi], 
                            self.seg_time_series[idx,:]
                        )[0,1]
                    )
            return chunk_results
        
        # Calculate optimal chunk size
        n_voxels = self.seg_time_series.shape[0]
        n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
        chunk_size = n_voxels // n_chunks
        
        # Parallel processing
        results = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(process_chunk)(i, chunk_size)
            for i in range(0, n_voxels, chunk_size)
        )
        
        # Combine results
        self.connectivity_matrix = np.concatenate(results)[:n_voxels]
        
        # Create and save connectivity map
        connectivity_map = nimsk.unmask(self.connectivity_matrix, self.seg_resampled)
        if dir_path:
            nib.save(
                connectivity_map,
                os.path.join(dir_path, f'sub{self.subject:03d}_subroi{sub_roi}_connectivitymap.nii')
            )

    def compute_connectivity_all(self, dir_path = None) -> None:
        ''' Compute the connectivity matrix between average tc in cluster and all other voxels in the segmented mask '''
        for i in range(self.n_clusters):
            self.compute_connectivity(i, dir_path)
    
    def compute_atlas_connectivity(self, atlas, atlas_ref, dir_path = None) -> None:
        ''' Compute the connectivity matrix between average tc in cluster and all other voxels in the segmented mask '''
        masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
        atlas_time_series = masker.fit_transform(test.func_img)
        corr_mat = np.corrcoef(self.averaged_timecourses, atlas_time_series.T)
        corr_mat = corr_mat[:self.n_clusters, self.n_clusters:]
        connectivity_df = atlas_ref.copy()
        for i in range(self.n_clusters):
            connectivity_df[f"connectivity_subroi{i+1}"] = corr_mat[i, :]
        if dir_path:
            connectivity_df.to_csv(os.path.join(dir_path, f'sub{self.subject:03d}_atlas_connectivity.csv'))



anat_src = "data/anat/sub-002_T1w.nii"
func_src = "data/func/sub-002_task-rest_bold.nii.gz"
roi_src = "data/mask_ROI/sub-002_hemi-L_FO_mask.nii"
seg_src = "data/segmentation/sub-002_space-orig_dseg.nii"
atlas_src = "data/atlas/atlas.nii"
atlas_img = nib.load(atlas_src)
atlas_data = atlas_img.get_fdata().astype(np.uint8)
lh_data = np.zeros(atlas_data.shape)
x = atlas_data.shape[0]
lh_data[:round(x/2), :, :] += atlas_data[:round(x/2) , :, :]
atlas_left = nib.Nifti1Image(lh_data, atlas_img.affine, dtype=np.uint8)



atlas_ref = pd.read_csv("data/atlas/atlas_ref.csv")



print("Processing subject 1")
test = Parcel(anat_src, func_src, roi_src, seg_src, 2)

print("Calculating zmap")
test.get_zmap(dir_path="results/distance_matrix")

print("Performing spectral clustering")
test.search_optimal_clusters(dir_path="results/silhouette_scores")
test.spectral_clustering(3, dir_path="results/mask_parceled")

print("Getting nifti labels")
test.get_nifti_labels(dir_path="results/mask_parceled")

print("Computing connectivity")
test.compute_connectivity_all(dir_path="results/connectivity")

print("Computing atlas connectivity")
test.compute_atlas_connectivity(atlas_left, atlas_ref, dir_path="results/connectivity")



# from nilearn import datasets, surface
# fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")
# img = nib.load("/Users/charles.verstraete/Documents/w0_OpFrontal/ParcelProject/RSfMRI-connectivity-based-parcalisation/results/connectivity/sub001_subroi0_connectivitymap.nii")

# a=1
# surf_data = surface.vol_to_surf(
#     img,
#     surf_mesh=fsaverage["infl_left"]
# )

# from nilearn import plotting


# chance = 0.5
# plotting.plot_surf_stat_map(
#     fsaverage["infl_left"],
#     surf_data,
#     view="lateral",
#     colorbar=True,
#     threshold=0.3,
#     bg_map=fsaverage["sulc_left"],
#     title="Accuracy map, left hemisphere",
# )



# plot_surf_contours(
#     roi_map=,
#     hemi="right",
#     labels=labels,
#     levels=regions_indices,
#     figure=figure,
#     legend=True,
#     colors=["g", "k"],
# )

# plotting.show()