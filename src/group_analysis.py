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

''' nifti package '''   
import nibabel as nib                                          
import nilearn.masking as nimsk
from nilearn.image import resample_img

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
from src.utils.morphometry_tools import *



roi_dir = "data/mask_ROI"
average_dir = "results/average"
os.makedirs(average_dir, exist_ok=True)

roi_files = sorted(os.listdir(roi_dir))

left_roi = [roi for roi in roi_files if roi.endswith('L_FO_mask.nii')]
right_roi = [roi for roi in roi_files if roi.endswith('R_FO_mask.nii')]

groupaverage_mask_from_dir(roi_dir, left_roi, os.path.join(average_dir, "leftFO_mask_average.nii"))
groupaverage_mask_from_dir(roi_dir, right_roi, os.path.join(average_dir, "rightFO_mask_average.nii"))

df = pd.read_csv("results/morphometry/morphometry.csv")

metrics = ['volume_mm3', 'com_y', 'com_z']


# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 10))

for idx, metric in enumerate(metrics):
    # Boxplot
        # Add individual points
    sns.stripplot(
        x='hemisphere', 
        y=metric, 
        data=df, 
        palette=['peachpuff', 'teal'], 
        alpha=1, 
        ax=axes[0, idx], 
        hue='hemisphere')
    
    sns.violinplot(
        x='hemisphere',
        y=metric, 
        data=df, 
        ax=axes[0, idx], 
        alpha=0.4, 
        hue='hemisphere', 
        palette=['peachpuff', 'teal'])

    # Perform t-test
    left = df[df['hemisphere'] == 'L'][metric].values
    right = df[df['hemisphere'] == 'R'][metric].values


    sns.regplot(
        x=left, 
        y=right, 
        data=df, 
        ax=axes[1, idx], 
        ci = 95,
        scatter_kws={'alpha': 0.3, 'color': 'black', 's' : 12},
        line_kws={'color': 'red', 'linewidth': 0.8})

    t_stat, p_val_ttest = stats.ttest_rel(left, right)
    corr, p_val_corr = stats.pearsonr(left, right)

        # Add title with p-value
    title = f'{metric}\nT-test: {t_stat:.2f}, p={p_val_ttest:.3e}' if p_val_ttest < 0.001 else f'{metric}\nT-test: {t_stat:.2f}, p={p_val_ttest:.3f}'
    axes[0, idx].set_title(title)

    title = f'Pearson corr: {corr:.2f}, p={p_val_corr:.3e}' if p_val_corr < 0.001 else f'Pearson corr: {corr:.2f}, p={p_val_corr:.3f}'
    axes[1, idx].set_title(title)

    
plt.tight_layout()
plt.show()


