# python 3.9.6
# -*- coding: utf-8 -*-
"""
parcel_analysis.py
================

Description:
    Perform group analysis on the parcelated brain regions

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


parcel_name_dict = {
    1 : 'anterior',
    2 : 'mid',
    3 : 'posterior'
}


# Chargement des labels relabellisés
relabel_dir = "results/group_analysis/relabelling"
relabel_left = pd.read_csv(os.path.join(relabel_dir, "relabelling_cluster_left.csv"), sep=";")
relabel_right = pd.read_csv(os.path.join(relabel_dir, "relabelling_cluster_right.csv"), sep=";")
relabel = {'left': relabel_left, 'right': relabel_right}

# Dossier contenant les résultats de la parcellisation
parcel_output_dir = "results/parcel_output"

# Liste des sujets
subject_file = sorted([d for d in os.listdir(parcel_output_dir) if d.startswith('sub-')])
subject_ids = [subject.split('-')[1] for subject in subject_file]
hemispheres = ['left', 'right']





connectivity_df = pd.DataFrame()

for subject_id in subject_ids:
    for hemi in hemispheres:
        # Chargement des labels relabellisés
        relabel_data = relabel[hemi]
        subject_num = int(subject_id)
        relabel_row = relabel_data[relabel_data['subject'] == subject_num]

        new_labels = relabel_row.values[0][-3:]

        # Chemin vers le fichier de connectivité
        connectivity_file = os.path.join(parcel_output_dir, f'sub-{subject_id}', 'connectivity', f'sub{subject_id}_HCP-MMP1-{hemi}_connectivity.csv')

        atlas_connectivity = pd.read_csv(connectivity_file)

        # Création d'un DataFrame temporaire pour stocker les colonnes réordonnées
        reordered_connectivity = atlas_connectivity.copy()
        for i, new_label in enumerate(new_labels):
            # Renommer les colonnes selon les nouveaux labels
            old_col = f'connectivity_subroi{new_label}'
            new_col = parcel_name_dict[i+1]
            reordered_connectivity.drop(columns=old_col, inplace=True)
            reordered_connectivity[new_col] = atlas_connectivity[old_col]

        # Ajout des informations
        reordered_connectivity['subject_id'] = subject_num
        reordered_connectivity['hemisphere'] = hemi

        # Ajout au DataFrame global
        connectivity_df = pd.concat([connectivity_df, reordered_connectivity])

connectivity_df.drop(columns=[col for col in connectivity_df.columns if 'Unnamed' in col], inplace=True)
connectivity_df.reset_index(drop=True, inplace=True)
connectivity_df

connectivity_cols = [col for col in connectivity_df.columns 
                    if col in ['anterior', 'mid', 'posterior']]

# Melt DataFrame
melted_connectivity = connectivity_df.melt(
    id_vars=['subject_id', 'hemisphere', 'ROI_n', 'ROI_glasser_full', 'ROI_glasser_1','ROI_glasser_2','Lobe','cortex'],
    value_vars=connectivity_cols,
    var_name='parcel',
    value_name='connectivity'
)


connectivity_df.to_csv('results/group_analysis/connectivity/group_HCP-MMP1_connectivity.csv', index=False)
melted_connectivity.to_csv('results/group_analysis/connectivity/group_HCP-MMP1_connectivity-melted.csv', index=False)


# Calcul de la moyenne des connectivités
average_connectivity = melted_connectivity.groupby(['hemisphere', 'parcel', 'ROI_glasser_1'])['connectivity'].mean().reset_index()

# Sauvegarde de la moyenne des connectivités
average_connectivity.to_csv('results/group_analysis/connectivity/group_HCP-MMP1_connectivity-average.csv', index=False)


# Seuil pour considérer une connectivité significative
threshold = 0.3

melted_connectivity['above_threshold'] = melted_connectivity['connectivity'] > threshold





probability_map = melted_connectivity.groupby(['ROI_n','hemisphere', 'parcel'])['above_threshold'].mean().reset_index()
probability_map.rename(columns={'above_threshold': 'probability'}, inplace=True)

# Sauvegarde de la probabilité map
probability_map.to_csv('results/group_analysis/connectivity/probability_map.csv', index=False)

from nilearn import plotting, datasets, surface


# Chargement de l'atlas en surface
fsaverage = datasets.fetch_surf_fsaverage()

atlas_left = surface.load_surf_data('data/atlas/lh.HCP-MMP1.annot')
atlas_right = surface.load_surf_data('data/atlas/rh.HCP-MMP1.annot')

fsaverage_name = "fsaverage5"

fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
fsaverage = datasets.fetch_surf_fsaverage(fsaverage_name)

coords_orig_left, _ = surface.load_surf_data(fsaverage_orig[f"pial_left"])
coords_target_left, _ = surface.load_surf_mesh(fsaverage[f"pial_left"])
resampled_atlas_left = surface_resample(atlas_left, coords_orig_left, coords_target_left)

coords_orig_right, _ = surface.load_surf_data(fsaverage_orig[f"pial_right"])
coords_target_right, _ = surface.load_surf_mesh(fsaverage[f"pial_right"])
resampled_atlas_right = surface_resample(atlas_right, coords_orig_right, coords_target_right)

atlas_dict = {'left': resampled_atlas_left, 'right': resampled_atlas_right}


# Create figure with new layout
fig, axes = plt.subplots(3, 4, figsize=(21, 12), 
                        subplot_kw={'projection': '3d'}, 
                        gridspec_kw={'wspace': 0.05, 'hspace': 0.1})
# Group by parcel first, then hemisphere
for parcel_idx, (parcel, parcel_data) in enumerate(probability_map.groupby('parcel')):
    for hemi_idx, (hemi, hemi_data) in enumerate(parcel_data.groupby('hemisphere')):
        
        # Get atlas data
        atlas = atlas_dict[hemi]
        atlas_filtered = np.zeros(atlas.shape)
        
        # Fill atlas with probabilities
        for _, row in hemi_data.iterrows():
            atlas_filtered[atlas == row['ROI_n']] = row['probability']
        
        vmin = 0
        vmax = 1
        
        # Plot lateral and medial views
        for view_idx, view in enumerate(['lateral', 'medial']):
            # Calculate column index: 2 * hemi_idx + view_idx
            col_idx = 2 * hemi_idx + view_idx
            
            display = plotting.plot_surf_roi(
                fsaverage[f'pial_{hemi}'],
                roi_map=atlas_filtered,
                view=view,
                hemi=hemi,
                bg_map=fsaverage[f'sulc_{hemi}'],
                cmap='hot',
                threshold=0.5,
                symmetric_cmap=False,  
                vmin=vmin,
                vmax=vmax,
                bg_on_data=True,
                axes=axes[parcel_idx, col_idx],
                title=f'{parcel.capitalize()} - {hemi} ({view})',
                colorbar=True,
                cbar_tick_format='%.2f'
            )
plt.tight_layout()
plt.savefig('results/group_analysis/figures/probability_map.png')
plt.show()

