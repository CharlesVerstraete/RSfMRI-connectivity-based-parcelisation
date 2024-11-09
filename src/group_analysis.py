# python 3.9.6
# -*- coding: utf-8 -*-
"""
group_analysis.py
================

Description:
    Perform group analysis on the parcelated brain regions and the connectivity

Dependencies:
    - numpy 1.23.5
    - pandas 1.3.3
    - nilearn 0.10.4
    - matplotlib 3.6.2
    - seaborn 0.11.2
    - scipy 1.7.3
    - statsmodels 0.13.0
    - sklearn 1.0

Author :
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""

# Import section
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

''' custom package '''
from utils.morphometry_tools import *



roi_dir = "data/mask_ROI"
average_dir = "results/average"
os.makedirs(average_dir, exist_ok=True)

roi_files = sorted(os.listdir(roi_dir))

left_roi = [roi for roi in roi_files if roi.endswith('L_FO_mask.nii')]
right_roi = [roi for roi in roi_files if roi.endswith('R_FO_mask.nii')]

groupaverage_mask(roi_dir, left_roi, os.path.join(average_dir, "leftFO_mask_average.nii"))
groupaverage_mask(roi_dir, right_roi, os.path.join(average_dir, "rightFO_mask_average.nii"))

df = pd.read_csv("results/morphometry/morphometry.csv")

metrics = ['volume_mm3', 'com_y', 'com_z']


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for idx, metric in enumerate(metrics):
    # Boxplot
    sns.boxplot(x='hemisphere', y=metric, data=df, ax=axes[0, idx])
    # Add individual points
    sns.stripplot(x='hemisphere', y=metric, data=df, 
                    color='black', alpha=0.3, ax=axes[0, idx])
    
    sns.regplot(x='L', y='R', data=df, ax=axes[1, idx])
    # Perform t-test
    left = df[df['hemisphere'] == 'L'][metric].values
    right = df[df['hemisphere'] == 'R'][metric].values

    t_stat, p_val = stats.ttest_rel(left, right)
    corr, pval = stats.pearsonr(left, right)

        # Add title with p-value
    axes[0, idx].set_title(f'{metric}\np={p_val:.3f}')
    axes[1, idx].set_title(f'{metric}\nCorrelation: {corr:.3f}, p={pval:.3f}')

    
plt.tight_layout()
plt.show()


