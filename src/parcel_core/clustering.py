import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import pandas as pd
from nilearn.masking import unmask
import nibabel as nib  

global EPSILON 
EPSILON = 1e-6

class SpectralClusteringAnalyzer:
    """Handles spectral clustering of ROI data."""
    def __init__(self, subject_id : int, hemi : str, roi_timeseries_mat : np.ndarray, seg_timeseries_mat : np.ndarray):
        self.subject = subject_id
        self.hemi = hemi
        self.roi_time_series = roi_timeseries_mat
        self.seg_time_series = seg_timeseries_mat
        self.roivox_distance = None
        self.labels = None
        self.n_clusters = None
        self._nOut = 0
        self._posOut = None

    def rm_nan(self, data: np.ndarray) -> np.ndarray:
        """
        Remove NaN values and return cleaned data with nan positions.
        """ 

        index_x = np.isnan(data).all(axis = 0)
        data_xnanless = np.delete(data, index_x, axis=1)
        index_y = np.isnan(data_xnanless).all(axis = 1)
        data_xynanless = np.delete(data_xnanless, index_y, axis=0)
        self._nOut = sum(index_y)
        self._posOut = np.where(index_y)[0]
        
        return data_xynanless

    def parallel_distance(self, mat_1: np.ndarray, mat_2: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """
        Calculate the distance matrix in parallel.
        """
        
        # Calculate chunk size
        n_chunks = os.cpu_count() if n_jobs == -1 else n_jobs
        n_features = len(mat_1)
        chunk_size = n_features // n_chunks
        chunks = []

        def process_chunk(start_idx: int) -> np.ndarray:
            end_idx = min(start_idx + chunk_size, n_features)
            return distance.cdist(
                mat_1[start_idx:end_idx],
                mat_2,
                metric="correlation"
            )
        
        chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk)(i) for i in range(0, n_features, chunk_size)
        )
                    
        return np.vstack(chunks)

    def get_zmap(self, dir_path = None, n_jobs = -1, precomputed = False) -> np.ndarray:
        ''' Calculate the z-map of the functional data '''
        if precomputed:
            zmap = np.load(os.path.join(dir_path, 'metrics', f'sub-{self.subject:03d}_hemi-{self.hemi}_roi-segmentation-zscore.npy'))
            self.roivox_distance = np.load(os.path.join(dir_path, 'metrics', f'sub-{self.subject:03d}_hemi-{self.hemi}_roi-intra-distance.npy'))
        else :
            distance_map = self.parallel_distance(self.roi_time_series, self.seg_time_series, n_jobs=n_jobs)
            correlation_map = 1 - distance_map
            correlation_map_nanless = self.rm_nan(correlation_map)
            correlation_map_nanless = np.clip(correlation_map_nanless, -1 + EPSILON, 1 - EPSILON) 
            zmap = np.arctanh(correlation_map_nanless)

            self.roivox_distance = self.parallel_distance(zmap, zmap, n_jobs=n_jobs)
            
            if dir_path:
                np.save(os.path.join(dir_path, 'metrics', f'sub-{self.subject:03d}_hemi-{self.hemi}_roi-segmentation-zscore.npy'), zmap)
                np.save(os.path.join(dir_path, 'metrics', f'sub-{self.subject:03d}_hemi-{self.hemi}_roi-intra-distance.npy'), self.roivox_distance)

    
    def perform_clustering(self, n_clusters: int):
        """ Perform spectral clustering on the data. """
        if self.roivox_distance is None:
            raise ValueError("Distance matrix not computed. Run get_zmap() first.")
        
        n_neighbors = np.sqrt(self.roi_time_series.shape[0]).astype(int)
        model = SpectralClustering(
            n_clusters = n_clusters, 
            affinity="precomputed_nearest_neighbors", 
            n_jobs = -1, 
            random_state = 6,
            n_neighbors = n_neighbors, 
            assign_labels='kmeans',).fit(self.roivox_distance)
        self.labels = model.labels_ + 1
        self.n_clusters = n_clusters
        if self._nOut != 0 :
            N = len(self.labels) + self._nOut
            full_labels = np.zeros(N, dtype=int)
            mask = np.ones(N, dtype=bool)
            mask[self._posOut] = False
            full_labels[mask] = self.labels
            self.labels = full_labels
            #self.labels = np.insert(self.labels, self._posOut, 0)

    def compute_silhouette(self, labels) -> float:
        return silhouette_score(self.roivox_distance, labels, metric='correlation')

    def search_optimal_clusters(self, dir_path : str, max_clusters : int = 10) -> int:
        """ Search for the optimal number of clusters. """
        scores = []
        n_neighbors = np.sqrt(self.roi_time_series.shape[0]).astype(int)
        for n_clusters in range(2, max_clusters + 1):
            model = SpectralClustering(
                n_clusters = n_clusters, 
                affinity="precomputed_nearest_neighbors", 
                n_jobs = -1, 
                random_state = 6,
                n_neighbors = n_neighbors, 
                assign_labels='kmeans').fit(self.roivox_distance)
            scores.append(self.compute_silhouette(model.labels_))
        df = pd.DataFrame({'n_clusters': range(2, max_clusters + 1), 'silhouette_score': scores})
        if dir_path:
            df.to_csv(os.path.join(dir_path, 'metrics', f'sub-{self.subject:03d}_hemi-{self.hemi}_silhouette-scores.csv'), index=False)

    def get_nifti_labelled(self, output_dir: str, target_img: nib.Nifti1Image):
        """ Save the labels as a Nifti file. """
        img = unmask(self.labels, target_img)
        nib.save(img, os.path.join(output_dir, 'cluster', f'sub{self.subject:03d}_hemi-{self.hemi}_FO_parcelated-{self.n_clusters}.nii'))
        for i in range(self.n_clusters):
            labels = np.zeros(self.labels.shape)
            labels[self.labels == i + 1] = 1
            img = unmask(labels, target_img)
            nib.save(img, os.path.join(output_dir, 'cluster', f'sub-{self.subject:03d}_hemi-{self.hemi}_parcel-{i + 1}.nii'))


    def get_average_timeseries(self, output_dir: str):
        """ Save the average time series for each cluster. """
        self.avg_tc = np.zeros((self.n_clusters, self.roi_time_series.shape[1]))
        for i in range(self.n_clusters):
            cluster_data = self.roi_time_series[self.labels == i + 1]
            np.save(os.path.join(output_dir, 'timeseries', f'sub{self.subject:03d}_hemi-{self.hemi}_parcel-{i + 1}-timeseries.npy'), cluster_data)
            self.avg_tc[i] = np.mean(cluster_data, axis=0)


