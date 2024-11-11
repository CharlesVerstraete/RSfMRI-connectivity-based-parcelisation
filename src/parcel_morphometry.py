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

def get_group_morphometry(parcel_output_dir: str, relabel : dict) -> pd.DataFrame:
    morphometry_df = pd.DataFrame()
    for i in range(1, 4):  # Supposons 3 clusters
        for hemi in hemispheres:
            new_cluster = relabel[hemi][parcel_name_dict[i]].values
            for idx, subject_id in enumerate(subject_ids):
                # Chemin vers les parcels relabellisés pour chaque sujet et hémisphère
                cluster_dir = os.path.join(parcel_output_dir, f'sub-{subject_id}', 'cluster')

                parcel_path = os.path.join(cluster_dir, f'sub-{subject_id}_hemi-{hemi}_parcel-{new_cluster[idx]}.nii')
                if os.path.exists(parcel_path):
                    # Calcul de la morphométrie
                    morpho_metrics = extract_morphometry(parcel_path)
                    # Ajout des informations supplémentaires
                    morpho_metrics['subject_id'] = subject_id
                    morpho_metrics['hemisphere'] = hemi
                    morpho_metrics['parcel'] = parcel_name_dict[i]
                    # Ajout au DataFrame
                    morphometry_df = pd.concat([morphometry_df, pd.DataFrame(morpho_metrics, index=[0])], axis=0)

    morphometry_df.reset_index(drop=True, inplace=True)
    return morphometry_df


def plot_violin(metric, label, morphometry_df, ax):
    """ Helper function for violin and strip plot with statistical annotations. """
    sns.violinplot(
        x='parcel', y=metric, 
        hue='hemisphere',
        data=morphometry_df,
        split=True,
        inner=None,
        palette=['peachpuff', 'teal'],
        alpha=0.4,
        ax=ax
    )
    sns.stripplot(
        x='parcel', y=metric,
        hue='hemisphere',
        data=morphometry_df,
        dodge=True,
        alpha=0.8,
        palette=['peachpuff', 'teal'],
        size=4,
        ax=ax,
        jitter=0.2
    )
    for parcel in morphometry_df['parcel'].unique():
        left = morphometry_df[(morphometry_df['parcel'] == parcel) & 
                              (morphometry_df['hemisphere'] == 'left')][metric]
        right = morphometry_df[(morphometry_df['parcel'] == parcel) & 
                               (morphometry_df['hemisphere'] == 'right')][metric]
        stat, pval = stats.ttest_rel(left, right)
        if pval < 0.05:
            ax.text(morphometry_df['parcel'].unique().tolist().index(parcel), 
                    morphometry_df[metric].max(),
                    f'*\np={pval:.3f}',
                    ha='center')
    ax.set_title(f'{label} Distribution')
    ax.set_ylabel(label)

def plot_radar(means_norm, ax, angles, metrics):
    """ Helper function for radar plot with normalized data. """
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for parcel in means_norm.index:
        values = np.concatenate((means_norm.loc[parcel].values, [means_norm.loc[parcel].values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {parcel}')
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(metrics.keys()))


def generate_distribution_plots(morphometry_df, metrics, output_dir):
    """ Generate distribution plots for each morphological metric. """
    fig = plt.figure(figsize=(20, 15))
    for idx, (metric, label) in enumerate(metrics.items()):
        ax = plt.subplot(3, 2, idx+1)
        plot_violin(metric, label, morphometry_df, ax)
        if idx > 3:
            ax.set_xlabel('Cluster')
        if idx == 0:
            ax.legend(title='Hemisphere')
        else:
            ax.get_legend().remove()
    plt.suptitle('Morphological Characteristics of Clusters', y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_distributions.pdf'))
    plt.close(fig)

def generate_correlation_matrix(morphometry_df, metrics, output_dir):
    """ Generate and save a heatmap of correlations between morphological measures. """
    plt.figure(figsize=(12, 10))
    correlation_matrix = morphometry_df[list(metrics.keys())].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
    plt.title('Correlation Between Morphological Measures')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_correlations.pdf'))
    plt.close()

def generate_radar_plots(morphometry_df, metrics, output_dir):
    """ Generate radar plots for each hemisphere, normalized by cluster. """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': 'polar'})
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    for idx, hemi in enumerate(['left', 'right']):
        data_hemi = morphometry_df[morphometry_df['hemisphere'] == hemi]
        means = data_hemi.groupby('parcel')[list(metrics.keys())].mean()
        min_val, max_val = means.min(), means.max()
        means_norm = 0.2 + (means - min_val) * 0.6 / (max_val - min_val)
        plot_radar(means_norm, axes[idx], angles, metrics)
        axes[idx].set_title(f'{hemi.capitalize()} Hemisphere')
    plt.legend(bbox_to_anchor=(1.2, 1.0))
    plt.suptitle('Normalized Morphological Properties by Cluster and Hemisphere', y=1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morphology_radar.pdf'))
    plt.close(fig)

def generate_statistical_summary(morphometry_df, metrics, output_dir):
    """ Generate a CSV file with statistical summary of t-tests between hemispheres. """
    stats_summary = pd.DataFrame()
    for metric in metrics:
        for parcel in morphometry_df['parcel'].unique():
            left = morphometry_df[(morphometry_df['parcel'] == parcel) & 
                                  (morphometry_df['hemisphere'] == 'left')][metric]
            right = morphometry_df[(morphometry_df['parcel'] == parcel) & 
                                   (morphometry_df['hemisphere'] == 'right')][metric]
            stat, pval = stats.ttest_rel(left, right)
            stats_summary = pd.concat([stats_summary, pd.DataFrame({
                'metric': [metric],
                'parcel': [parcel],
                'statistic': [stat],
                'p_value': [pval]
            })])
    stats_summary.to_csv(os.path.join(output_dir, 'morphology_statistics.csv'), index=False)

def visualize_morphology(morphometry_df: pd.DataFrame, output_dir: str):
    """ Main function to orchestrate the generation of morphological visualizations. """
    
    metrics = {
        'volume_mm3': 'Volume (mm³)',
        'com_y': 'Y Center of Mass (mm)',
        'com_z': 'Z Center of Mass (mm)',
        'length_mm': 'Length (mm)',
        'width_mm': 'Width (mm)',
        'height_mm': 'Height (mm)'
    }
    
    generate_distribution_plots(morphometry_df, metrics, output_dir)
    generate_correlation_matrix(morphometry_df, metrics, output_dir)
    generate_radar_plots(morphometry_df, metrics, output_dir)
    generate_statistical_summary(morphometry_df, metrics, output_dir)



if __name__ == "__main__":
    # Dictionnaire pour les noms des parcelles
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

    morphometry_df = get_group_morphometry(parcel_output_dir, relabel)
    morphometry_df.to_csv('results/group_analysis/morphometry/group-parcels-FO_morphometry.csv', index=False)

    visualize_morphology(morphometry_df, 'results/group_analysis/morphometry')


