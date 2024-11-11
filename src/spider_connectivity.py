# python 3.9.6
# -*- coding: utf-8 -*-
"""
spider_connectivity.py
================

Description:
    Perform spider chart analysis on the connectivity data

Dependencies:
    - nibabel 3.2.1
    - numpy 1.23.5
    - nilearn 0.10.4
    - matplotlib 3.6.2  
    - pandas 1.3.3

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




melted_connectivity = pd.read_csv('results/group_analysis/connectivity/group_HCP-MMP1_connectivity-melted.csv')
ref_atlas = pd.read_csv('data/atlas/HCP-MMP1_labels.csv', sep=';')

melted_connectivity

# create a spider chart for anterior region

# get the connectivity values for the anterior region
anterior_connectivity = melted_connectivity[melted_connectivity['parcel'] == 'anterior']


parcel_data, mean_conn, all_rois = prepare_connectivity_data(melted_connectivity, 'anterior')


all_rois = sorted(melted_connectivity['ROI_glasser_full'].unique())


def prepare_connectivity_data(df, parcel):
    """Prepare connectivity data for a given parcel."""
    parcel_data = df[df['parcel'] == parcel]
    all_rois = sorted(df['ROI_glasser_full'].unique())
    
    mean_conn = parcel_data.groupby(
        ['hemisphere', 'ROI_glasser_full']
    )['connectivity'].mean().reset_index()

    top_rois = (mean_conn.groupby('hemisphere')
                .apply(lambda x: x.nlargest(10, 'connectivity'))
                .reset_index(drop=True))
    
    return parcel_data, mean_conn, top_rois["ROI_glasser_full"].unique().tolist()

def compute_statistics(parcel_data, all_rois):
    """Compute statistical differences between hemispheres."""
    significant_rois = []
    for roi in all_rois:
        left = parcel_data[(parcel_data['hemisphere'] == 'left') & 
                          (parcel_data['ROI_glasser_full'] == roi)]['connectivity']
        right = parcel_data[(parcel_data['hemisphere'] == 'right') & 
                           (parcel_data['ROI_glasser_full'] == roi)]['connectivity']
        
        if len(left) > 0 and len(right) > 0:
            _, pval = stats.ttest_ind(left, right)
            if pval < 0.05:
                significant_rois.append((roi, pval))
    return significant_rois

def create_radar_plot(ax, mean_conn, all_rois, significant_rois, parcel, colors):
    """Create radar plot for one parcel."""
    # Prepare data
    left_data = mean_conn[mean_conn['hemisphere'] == 'left'].set_index('ROI_glasser_full')['connectivity']
    right_data = mean_conn[mean_conn['hemisphere'] == 'right'].set_index('ROI_glasser_full')['connectivity']
    
    angles = np.linspace(0, 2*np.pi, len(all_rois), endpoint=False)
    left_values = [left_data.get(roi, 0) for roi in all_rois]
    right_values = [right_data.get(roi, 0) for roi in all_rois]
    
    # Close the polygons
    angles = np.concatenate((angles, [angles[0]]))
    left_values = np.concatenate((left_values, [left_values[0]]))
    right_values = np.concatenate((right_values, [right_values[0]]))
    
    # Plot
    ax.plot(angles, left_values, 'o-', linewidth=2, color=colors['left'], label='Gauche')
    ax.plot(angles, right_values, 'o-', linewidth=2, color=colors['right'], label='Droit')
    ax.fill(angles, left_values, alpha=0.25, color=colors['left'])
    ax.fill(angles, right_values, alpha=0.25, color=colors['right'])
    
    # Add significance markers
    for roi, pval in significant_rois:
        roi_idx = all_rois.index(roi)
        marker = '*' if pval < 0.05 else '**' if pval < 0.01 else '***'
        ax.text(angles[roi_idx], max(left_values[roi_idx], right_values[roi_idx]) * 1.1, 
                marker, ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(all_rois, size=8)
    
    # Rotate and align the tick labels so they look better
    for label in ax.get_xticklabels():
        label.set_rotation(45)
    
    # Add a legend
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    # Add a title
    ax.set_title(f'{parcel.capitalize()}', pad=20)
    
    # Add gridlines
    ax.grid(True, alpha=0.3)

def plot_connectivity_comparison(df):
    """Main function to create connectivity comparison plots."""
    colors = {'left': '#1f77b4', 'right': '#ff7f0e'}  # Bleu pour gauche, Orange pour droit
    
    # Create three subplots
    fig, axes = plt.subplots(1, 3, figsize=(17, 10), 
                            subplot_kw=dict(projection='polar'))
    
    # Create plots for each parcel
    for idx, parcel in enumerate(['anterior', 'mid', 'posterior']):
        parcel_data, mean_conn, all_rois = prepare_connectivity_data(df, parcel)
        significant_rois = compute_statistics(parcel_data, all_rois)
        create_radar_plot(axes[idx], mean_conn, all_rois, significant_rois, parcel, colors)
    
    # Add a main title
    plt.suptitle('Comparaison des Hémisphères par Parcelle\n* p<0.05, ** p<0.01, *** p<0.001', 
                 y=0.95, fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

# Utiliser la fonction (en supposant que votre DataFrame s'appelle melted_connectivity)
fig = plot_connectivity_comparison(melted_connectivity)
plt.show()
