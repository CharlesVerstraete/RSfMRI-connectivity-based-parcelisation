# python 3.9.6
# -*- coding: utf-8 -*-
"""
parcel.py
=========

Description:
    Main class object for RS-fMRI connectivity based parcellation.

Author:
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""

# Import libraries

from parcel_core import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nilearn.input_data import NiftiLabelsMasker
import nibabel as nib
import pandas as pd

matplotlib.use('Qt5Agg')

#from src.parcel_core import ConnectivityAnalyzer

class Parcel:
    """Main class coordinating the parcellation analysis."""
    def __init__(self, subject_id: int, hemi : str, data_paths: dict, output_dir: str):
        self.subject_id = subject_id
        self.hemi = hemi
        self.io = IOManager(output_dir)
        
        # Initialize components
        self.images = ImageLoader(data_paths['anat'], data_paths['func'])
        self.roi = ROIProcessor(data_paths['roi'], self.images.func_img)
        self.segmentation = ROIProcessor(data_paths['seg'], self.images.func_img)
        self.segmentation.resampled_data = (self.segmentation.resampled_data - self.roi.resampled_data > 0).astype(np.uint8)
        self.clustering = None
        self.connectivity = None

    def init_clustering(self):
        roi_timeseries_mat = self.roi.extract_time_series(self.images.func_img)
        seg_timeseries_mat = self.segmentation.extract_time_series(self.images.func_img)
        self.clustering = SpectralClusteringAnalyzer(self.subject_id, self.hemi,  roi_timeseries_mat.T, seg_timeseries_mat.T)
        
    def search_optimal_clusters(self, precomputed = False, max_clusters: int = 10):
        """Search for the optimal number of clusters."""
        if self.clustering is None:
            raise ValueError("Clustering object not initialized. Run init_clustering() first.")
        if self.clustering.roivox_distance is None:
            if precomputed :
                self.clustering.get_zmap(self.io.output_dir, precomputed=precomputed)
            else:
                print("Distance matrix not computed. Calculating zmap...")
                self.clustering.get_zmap(self.io.output_dir)
        self.clustering.search_optimal_clusters(self.io.output_dir, max_clusters)
    
    def perform_clustering(self, n_clusters: int, n_jobs: int = -1, zmap_precomputed: bool = False):
        """Perform spectral clustering."""
        if self.clustering.roivox_distance is None:
            print("Distance matrix not computed. Calculating zmap...")
            self.clustering.get_zmap(self.io.output_dir, n_jobs = n_jobs, precomputed=zmap_precomputed)
        self.clustering.perform_clustering(n_clusters)
        self.clustering.get_nifti_labelled(self.io.output_dir, self.roi.resampled_img)    
        self.clustering.get_average_timeseries(self.io.output_dir)

    def init_connectivity(self):
        if self.clustering is None:
            raise ValueError("Clustering object not initialized. Run init_clustering() first.")
        self.connectivity = ConnectivityAnalyzer(
            self.subject_id, 
            self.hemi,
            self.clustering.seg_time_series, 
            self.clustering.avg_tc, 
            self.segmentation.resampled_img, 
        )

    def perform_wholebrain_connectivity(self, n_jobs: int = -1):
        """Compute connectivity matrix."""
        for i in range(self.clustering.n_clusters):
            self.connectivity.compute_wholebrain_connectivity(i, self.io.output_dir,  n_jobs=n_jobs)
    
    def perform_atlas_connectivity(self, atlas_img: nib.Nifti1Image, atlas_df: pd.DataFrame):
        """Compute connectivity with atlas."""
        masker = NiftiLabelsMasker(labels_img=atlas_img)
        atlas_time_series = masker.fit_transform(self.images.func_img)
        self.connectivity.compute_atlas_connectivity(atlas_time_series, atlas_df, self.io.output_dir)






# from nilearn.masking import unmask

# connectivity_map = unmask(
#     connect.connectivity_matrix, 
#     connect.seg_mask
# )


# import matplotlib.pyplot as plt

# from nilearn import plotting, surface, datasets
# from utils.plot_tools import surface_resample


# connectivity_df = pd.read_csv("results/parcel_output/sub-001/sub001_atlas_connectivity.csv")
# connectivity_df

# lh_atlas_src = "data/atlas/lh.HCP-MMP1.annot"
# lh_atlas = surface.load_surf_data(lh_atlas_src)
# lh_atlas_filtered = np.zeros(lh_atlas.shape)

# fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
# fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage3")

# coords_fsaverage, _ = surface.load_surf_mesh(fsaverage["pial_left"])
# coords_fsaverage5, _ = surface.load_surf_mesh(fsaverage5["pial_left"])

# lh_atlas_fsaverage5 = surface_resample(lh_atlas, coords_fsaverage, coords_fsaverage5)


# views = ['lateral', 'medial']


# # Create figure with subplots
# fig, axes = plt.subplots(3, 2, figsize=(17, 12), 
#                         subplot_kw={'projection': '3d'}, 
#                         gridspec_kw={'wspace': 0.1, 'hspace': 0.1, 'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1})

# #connectivity_df
# # In the plotting loop
# for i in range(3):
#     for j, view in enumerate(views):
#         top_ids = connectivity_df[np.abs(connectivity_df[f"connectivity_subroi{i+1}"]) > 0.5]
#         # Create connectivity map
#         lh_atlas_filtered = np.zeros(lh_atlas_fsaverage5.shape)
#         for _, row in top_ids.iterrows():
#             lh_atlas_filtered[lh_atlas_fsaverage5 == row['ROI_n']] = row[f'connectivity_subroi{i+1}']
        
#         vmin = top_ids[f'connectivity_subroi{i+1}'].min()
#         vmax = top_ids[f'connectivity_subroi{i+1}'].max()
#         display = plotting.plot_surf_roi(
#             fsaverage5['pial_left'],
#             roi_map=lh_atlas_filtered,
#             view=view,
#             hemi='left',
#             bg_map=fsaverage5['sulc_left'],
#             cmap='viridis',
#             vmin=-1,
#             vmax=1,
#             bg_on_data=True,
#             axes=axes[i, j],
#             title=f'Sub-ROI {i+1} - {view}',
#             colorbar=True,
#             cbar_tick_format='%.2f'
#         )

#     top_regions = top_ids['ROI_glasser_2'].unique()
#     axes[i, 0].text2D(-0.25, 0.95, 
#                         f"Top regions:\n" + "\n".join(top_regions),
#                         transform=axes[i, 0].transAxes,
#                         fontsize=6,
#                         verticalalignment='top')

# plt.tight_layout()
# plt.show()






# connectivity_df










# for i in range(3):  # For each sub-ROI

#     # Plot each view
#     for j, view in enumerate(views):
#         plotting.plot_surf_roi(
#             fsaverage5['pial_left'],
#             roi_map=lh_atlas_filtered,
#             view=view,
#             hemi='left',
#             bg_map=fsaverage5['sulc_left'],
#             cmap='viridis',
#             bg_on_data=True,
#             axes=axes[i, j],
#             title=f'Sub-ROI {i+1} - {view}',
#         )

# plt.colorbar()
# plt.tight_layout()
# plt.show()

# import os













# fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage5")
# stat_map2 = surface.vol_to_surf(connectivity_map, fsaverage5["pial_left"])

# fig, ax = plt.subplots(2, 1, projection="3d", figsize=(10, 10))

# plotting.plot_surf_stat_map(
#     fsaverage5["pial_left"],
#     stat_map,
#     view="lateral",
#     colorbar=True,
#     threshold=0.3,
#     bg_map=fsaverage5["sulc_left"],
#     title="Connectivity map, left hemisphere",
#     axes=ax[0]
# )



# plotting.plot_surf_stat_map(
#     fsaverage5["pial_left"],
#     stat_map2,
#     view="lateral",
#     colorbar=True,
#     threshold=0.3,
#     bg_map=fsaverage5["sulc_left"],
#     title="Connectivity map, left hemisphere",
#     axes=ax[1]
# )




# print("Searching optimal clusters")
# roi_timeseries_mat = parcel.roi.extract_time_series(parcel.images.func_img)
# seg_timeseries_mat = parcel.segmentation.extract_time_series(parcel.images.func_img)
# cluster = SpectralClusteringAnalyzer(subject_id, roi_timeseries_mat.T, seg_timeseries_mat.T)
# cluster.get_zmap(output_dir)
# cluster.search_optimal_clusters(output_dir)


# print("Performing clustering")
# cluster.perform_clustering(3)
# cluster.get_nifti_labels(output_dir, parcel.roi.resampled_img)
# cluster.get_average_timeseries(output_dir)

# ConnectivityAnalyzer(seg_timeseries_mat, cluster.avg_tc, parcel.segmentation.resampled_img, subject_id)























# # np_signal_idx = np.where(seg_timeseries_mat.sum(axis=0) == 0)[0]
# # seg_timeseries_mat = np.delete(seg_timeseries_mat, np_signal_idx, axis=1)


# # n_jobs = 6

# # import os
# # from scipy.spatial import distance
# # from joblib import Parallel, delayed
# # from tqdm import tqdm

# # mat_1 = roi_timeseries_mat.T
# # mat_2 = seg_timeseries_mat.T


# # n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
# # n_features = len(mat_1)

# # chunk_size = n_features // n_chunks


# # def process_chunk(mat_1, mat_2, start_idx, end_idx) -> np.ndarray:
# #     return distance.cdist(
# #         mat_1[start_idx:end_idx],
# #         mat_2,
# #         metric="correlation"
# #     )

# # # Parallel processing with progress bar
# # # with tqdm(total=n_features, desc='Computing distances') as pbar:

# # chunks = []
# # chunks = Parallel(n_jobs=n_jobs)(
# #     delayed(process_chunk)(mat_1, mat_2, i, i + chunk_size) for i in range(0, n_features, chunk_size)
# # )

# # distance_map = np.vstack(chunks)         
# # cmap = 1 - distance_map
# # cmap[cmap >= 1] = 0.9999999999999999
# # zmap = np.arctanh(cmap)


# # np.any(np.isnan(zmap))
# # np.any(~np.isfinite(zmap))


# # test_distance = distance.cdist(mat_1, mat_2[-60000:], metric="correlation")
# # zmap = np.arctanh(1 - test_distance)



# # plt.imshow(zmap, aspect='auto', origin='lower', cmap='viridis', vmin=-1, vmax=1)
# # plt.colorbar()
# # plt.show()



# # cluster = SpectralClusteringAnalyzer(roi_timeseries_mat.T, seg_timeseries_mat.T)
# # correlation_map = cluster.parallel_distance(cluster.roi_time_series, cluster.seg_time_series, n_jobs=-1)
# # correlation_map_nanless = cluster.rm_nan(correlation_map)

# # zmap = np.arctanh(1 - correlation_map_nanless)

# # plt.imshow(zmap[np.isfinite(zmap)], aspect='auto', origin='lower')
# # plt.show()




# # zmap.max() 
# # zmap[np.isfinite(zmap)]

# # def preprocess_matrix(mat):
# #     max_val = np.max(np.abs(zmap[np.isfinite(zmap)]))
# #     if np.any(np.isnan(mat)):
# #         mat = np.nan_to_num(mat, 0)  # Replace NaN with 0
# #     if np.any(~np.isfinite(mat)):
# #         mat = np.clip(mat, -1e10, 1e10)  # Clip infinite values
# #     return mat

# # mat_1 = preprocess_matrix(zmap)
# # mat_2 = preprocess_matrix(mat_2)



# # plt.imshow(correlation_map, aspect='auto', origin='lower')
# # plt.show()



# # plt.plot(test)
# # plt.show()




# # plt.plot(test.T)
# # plt.show()




# # parcel.run_analysis(3)










# # atlas_ref = pd.read_csv("data/atlas/atlas_ref.csv")



# # print("Processing subject 1")
# # test = Parcel(anat_src, func_src, roi_src, seg_src, 2)

# # print("Calculating zmap")
# # test.get_zmap(dir_path="results/distance_matrix")

# # print("Performing spectral clustering")
# # test.search_optimal_clusters(dir_path="results/silhouette_scores")
# # test.spectral_clustering(3, dir_path="results/mask_parceled")

# # print("Getting nifti labels")
# # test.get_nifti_labels(dir_path="results/mask_parceled")

# # print("Computing connectivity")
# # test.compute_connectivity_all(dir_path="results/connectivity")

# # print("Computing atlas connectivity")
# # test.compute_atlas_connectivity(atlas_left, atlas_ref, dir_path="results/connectivity")



# # from nilearn import datasets, surface
# # fsaverage = datasets.fetch_surf_fsaverage("fsaverage6")
# # img = nib.load("/Users/charles.verstraete/Documents/w0_OpFrontal/ParcelProject/RSfMRI-connectivity-based-parcalisation/results/connectivity/sub001_subroi0_connectivitymap.nii")

# # a=1
# # surf_data = surface.vol_to_surf(
# #     img,
# #     surf_mesh=fsaverage["infl_left"]
# # )

# # from nilearn import plotting


# # chance = 0.5
# # plotting.plot_surf_stat_map(
# #     fsaverage["infl_left"],
# #     surf_data,
# #     view="lateral",
# #     colorbar=True,
# #     threshold=0.3,
# #     bg_map=fsaverage["sulc_left"],
# #     title="Accuracy map, left hemisphere",
# # )



# # plot_surf_contours(
# #     roi_map=,
# #     hemi="right",
# #     labels=labels,
# #     levels=regions_indices,
# #     figure=figure,
# #     legend=True,
# #     colors=["g", "k"],
# # )

# # plotting.show()




# # class Parcel:
# #     def __init__(self, 
# #                  anat_src : str, 
# #                  func_src : str, 
# #                  roi_src : str, 
# #                  seg_src : str, 
# #                  subject : int,
# #                  zmap_computed : bool  = False,
# #                  output_dir : str = None,
# #                  ) -> None:
# #         ''' Initialize the Parcel object '''
# #         self.subject = subject

# #         self.anat_img = nib.load(anat_src)
# #         self.func_img = nib.load(func_src)
# #         self.roi_img = nib.load(roi_src)
# #         self.seg_img = nib.load(seg_src)

# #         self.anat_data = self.anat_img.get_fdata()
# #         self.func_data = self.func_img.get_fdata()
# #         self.anat_shape = self.anat_data.shape
# #         self.anat_affine = self.anat_img.affine
# #         self.func_shape = self.func_data.shape
# #         self.func_affine = self.func_img.affine

# #         self.roi_data = self.roi_img.get_fdata().astype(int)
# #         self.seg_data = self.seg_img.get_fdata().astype(int)

# #         # ROI mask processing
# #         self.roi_resampled = resample_img(
# #             self.roi_img, 
# #             target_affine=self.func_affine, 
# #             target_shape=self.func_shape[:3], 
# #             interpolation='nearest')
# #         self.roi_data_resampled = self.roi_resampled.get_fdata()
# #         self.roi_data_resampled = (self.roi_data_resampled > 0).astype(np.uint8)
# #         self.roi_resampled = nib.Nifti1Image(
# #             self.roi_data_resampled, 
# #             self.func_affine, 
# #             dtype=np.uint8
# #         )

# #         # Segmentation mask processing
# #         self.seg_resampled = resample_img(
# #             self.seg_img, 
# #             target_affine=self.func_affine, 
# #             target_shape=self.func_shape[:3],
# #             interpolation='nearest')
# #         self.seg_data_resampled = self.seg_resampled.get_fdata()
# #         self.seg_data_resampled = (self.seg_data_resampled - self.roi_data_resampled > 0).astype(np.uint8)
# #         self.seg_resampled = nib.Nifti1Image(
# #             self.seg_data_resampled, 
# #             self.func_affine,
# #             dtype=np.uint8
# #         )

# #         self.roi_time_series = nimsk.apply_mask(self.func_img, self.roi_resampled).T
# #         self.seg_time_series = nimsk.apply_mask(self.func_img, self.seg_resampled).T

# #         self.roi_nvoxels = self.roi_time_series.shape[1]
# #         self.roivox_distance = np.zeros((self.roi_nvoxels, self.roi_nvoxels))
# #         self.n_clusters = 0
# #         self.labels = []
# #         self.centroids = []
# #         self.silhouette_score = 0
# #         self.n_out = 0
# #         self.position_out = []


# #     def rm_nan(self, data : np.ndarray) -> np.ndarray:
# #         ''' Remove NaN values from the data and store position'''

# #         index_x = np.isnan(data).all(axis = 0)
# #         data_xnanless = np.delete(data, index_x, axis=1)
# #         index_y = np.isnan(data_xnanless).all(axis = 1)
# #         data_xynanless = np.delete(data_xnanless, index_y, axis=0)
# #         self._nOut = sum(index_y)
# #         self._posOut = np.where(index_y)[0]
        
# #         return data_xynanless

# #     def get_zmap(self, dir_path = None, n_jobs = -1) -> np.ndarray:
# #         ''' Calculate the z-map of the functional data '''
# #         def chunk_correlations(start_idx, end_idx):
# #             chunk = distance.cdist(
# #                 self.roi_time_series[start_idx:end_idx], 
# #                 self.seg_time_series, 
# #                 metric="correlation"
# #             )
# #             return chunk
# #         # Calculate chunk size
# #         n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
# #         chunk_size = len(self.roi_time_series) // n_chunks

# #         # Parallel correlation computation
# #         chunks = Parallel(n_jobs=n_jobs)(
# #             delayed(chunk_correlations)(i, i + chunk_size)
# #             for i in range(0, len(self.roi_time_series), chunk_size)
# #         )

# #         correlation_map = np.vstack(chunks)
# #         correlation_map_nanless = self.rm_nan(correlation_map)
# #         zmap = np.arctanh(1 - correlation_map_nanless)
# #         # Compute ROI voxel distances in parallel
# #         roivox_distance = Parallel(n_jobs=n_jobs)(
# #             delayed(distance.cdist)(zmap[i:i+chunk_size], zmap, metric="correlation")
# #             for i in range(0, len(zmap), chunk_size)
# #         )
# #         self.roivox_distance = np.vstack(roivox_distance)
# #         if dir_path:
# #             np.save(os.path.join(dir_path, f'sub{self.subject:03d}-zmap.npy'), zmap)
# #             np.save("roivox_distance.npy", self.roivox_distance)

# #     def spectral_clustering(self, n_clusters : int, dir_path = None) -> None:
# #         ''' Perform spectral clustering on the z-map '''
# #         self.n_clusters = n_clusters
# #         spectral = SpectralClustering(n_clusters= self.n_clusters, affinity="precomputed_nearest_neighbors", n_jobs = -1, random_state = 6)
# #         self.labels = spectral.fit_predict(self.roivox_distance)
# #         self.averaged_timecourses = np.array([np.mean(self.roi_time_series[self.labels == i], axis = 0) for i in range(n_clusters)])
# #         self.silhouette_score = silhouette_score(self.roivox_distance, self.labels, metric = "correlation")
# #         if dir_path:
# #             np.save(os.path.join(dir_path, f'sub{self.subject:03d}-{self.n_clusters}clusters-labels.npy'), self.labels)
# #             np.save(os.path.join(dir_path, f'sub{self.subject:03d}-{self.n_clusters}clusters-averaged_timecourses.npy'), self.averaged_timecourses)
    
# #     def search_optimal_clusters(self, max_clusters : int = 10, dir_path = None) -> None:
# #         ''' Search for the optimal number of clusters '''
# #         silhouette_scores = []
# #         for n_clusters in range(2, max_clusters + 1):
# #             self.spectral_clustering(n_clusters)
# #             silhouette_scores.append(self.silhouette_score)
# #         df = pd.DataFrame({"n_clusters": range(2, max_clusters + 1), "silhouette_score": silhouette_scores})
# #         if dir_path:
# #             df.to_csv(os.path.join(dir_path, f'sub{self.subject:03d}-silhouette_scores.csv'))
    
# #     def get_nifti_labels(self, dir_path = None) -> None:
# #         ''' Get the nifti labels '''
# #         self.labels += 1
# #         if self.n_out != 0 :
# #             self.labels = np.insert(self.labels, self._posOut, -1)
# #         roi_parceled = nimsk.unmask(self.labels, self.roi_resampled)
# #         roi_parceled_data = roi_parceled.get_fdata()
# #         for i in range(self.n_clusters):
# #             roi_parceled_data_copy = roi_parceled_data.copy()
# #             roi_parceled_data_copy[roi_parceled_data_copy != i + 1] = 0
# #             subroi_parcel = nib.Nifti1Image(roi_parceled_data_copy, self.func_affine, dtype=np.uint8)
# #             if dir_path:
# #                 nib.save(subroi_parcel, os.path.join(dir_path, f'sub{self.subject:03d}-parcel-{i+1}.nii'))
# #         if dir_path:
# #             nib.save(roi_parceled, os.path.join(dir_path, f'sub{self.subject:03d}-parcel.nii'))


# #     def compute_connectivity(self, sub_roi, dir_path=None, n_jobs=-1) -> None:
# #         """Compute connectivity matrix in parallel."""
        
# #         def process_chunk(chunk_start, chunk_size):
# #             chunk_results = np.zeros(chunk_size)
# #             for i in range(chunk_size):
# #                 idx = chunk_start + i
# #                 if idx < self.seg_time_series.shape[0]:
# #                     chunk_results[i] = np.arctanh(
# #                         np.corrcoef(
# #                             self.averaged_timecourses[sub_roi], 
# #                             self.seg_time_series[idx,:]
# #                         )[0,1]
# #                     )
# #             return chunk_results
        
# #         # Calculate optimal chunk size
# #         n_voxels = self.seg_time_series.shape[0]
# #         n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
# #         chunk_size = n_voxels // n_chunks
        
# #         # Parallel processing
# #         results = Parallel(n_jobs=n_jobs, verbose=1)(
# #             delayed(process_chunk)(i, chunk_size)
# #             for i in range(0, n_voxels, chunk_size)
# #         )
        
# #         # Combine results
# #         self.connectivity_matrix = np.concatenate(results)[:n_voxels]
        
# #         # Create and save connectivity map
# #         connectivity_map = nimsk.unmask(self.connectivity_matrix, self.seg_resampled)
# #         if dir_path:
# #             nib.save(
# #                 connectivity_map,
# #                 os.path.join(dir_path, f'sub{self.subject:03d}_subroi{sub_roi}_connectivitymap.nii')
# #             )

# #     def compute_connectivity_all(self, dir_path = None) -> None:
# #         ''' Compute the connectivity matrix between average tc in cluster and all other voxels in the segmented mask '''
# #         for i in range(self.n_clusters):
# #             self.compute_connectivity(i, dir_path)
    
# #     def compute_atlas_connectivity(self, atlas, atlas_ref, dir_path = None) -> None:
# #         ''' Compute the connectivity matrix between average tc in cluster and all other voxels in the segmented mask '''
# #         masker = NiftiLabelsMasker(labels_img=atlas_img, standardize=True)
# #         atlas_time_series = masker.fit_transform(test.func_img)
# #         corr_mat = np.corrcoef(self.averaged_timecourses, atlas_time_series.T)
# #         corr_mat = corr_mat[:self.n_clusters, self.n_clusters:]
# #         connectivity_df = atlas_ref.copy()
# #         for i in range(self.n_clusters):
# #             connectivity_df[f"connectivity_subroi{i+1}"] = corr_mat[i, :]
# #         if dir_path:
# #             connectivity_df.to_csv(os.path.join(dir_path, f'sub{self.subject:03d}_atlas_connectivity.csv'))

