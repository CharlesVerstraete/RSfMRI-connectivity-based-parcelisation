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

''' nifti package '''   
import nibabel as nib                                          
import nilearn.masking as nimsk
from nilearn.image import resample_img

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

        self.roi_data = self.roi_img.get_fdata().astype(int)
        self.seg_data = self.seg_img.get_fdata().astype(int)
        self.roi_data_resampled = resample_img(self.roi_img, target_affine=self.func_img.affine, target_shape=self.func_data.shape).get_fdata().astype(int)
        self.seg_data_resampled = resample_img(self.seg_img, target_affine=self.func_img.affine, target_shape=self.func_data.shape).get_fdata().astype(int)
        self.seg_data_resampled = np.abs(self.seg_data_resampled - self.roi_data_resampled).astype(int)
        
        self.roi_time_series = nimsk.apply_mask(self.func_img, self.roi_img).T
        self.seg_time_series = nimsk.apply_mask(self.func_img, self.seg_img).T

        self.anat_shape = self.anat_data.shape
        self.anat_affine = self.anat_img.affine
        self.func_shape = self.func_data.shape
        self.func_affine = self.func_img.affine
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

    def get_zmap(self, dir_path = None) -> np.ndarray:
        ''' Calculate the z-map of the functional data '''
        correlation_map = distance.cdist(self.roi_time_series, self.seg_time_series, metric = "correlation")
        correlation_map_nanless = self.rm_nan(correlation_map)
        zmap = np.arctanh(1-correlation_map_nanless)
        self.roivox_distance = distance.cdist(zmap, zmap, metric = "correlation")
        if dir_path:
            np.save(os.path.join(dir_path, f'sub{self.subject:03d}-zmap.npy'), zmap)
            np.save("roivox_distance.npy", self.roivox_distance)

    




