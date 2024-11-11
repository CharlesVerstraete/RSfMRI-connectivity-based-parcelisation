# python 3.9.6
# -*- coding: utf-8 -*-
"""
optimal_score_analysis.py
================

Description:
    Quick script for optimal number of clusters

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


''' stat package '''   
import pandas as pd

''' visualisation package '''   
import matplotlib.pyplot as plt
import seaborn as sns

import os 


def get_subject_ids(parcel_output_dir):
    subject_files = os.listdir(parcel_output_dir)
    if '.DS_Store' in subject_files:
        subject_files.remove('.DS_Store')
    subject_files.sort()
    subject_ids = [x.split('-')[-1] for x in subject_files]
    return subject_ids, subject_files

def get_group_score_df(parcel_output_dir):
    group_score_df = pd.DataFrame()
    for subject_id in subject_ids:
        for hemi in ['left', 'right']:
            subject_dir = os.path.join(parcel_output_dir, f'sub-{subject_id}')
            score_file = os.path.join(subject_dir, 'metrics', f'sub-{subject_id}_hemi-{hemi}_silhouette-scores.csv')
            score_df = pd.read_csv(score_file)
            score_df["subject_id"] = int(subject_id)
            score_df["hemi"] = hemi
            group_score_df = pd.concat([group_score_df, score_df])
    group_score_df.reset_index(drop=True, inplace=True)
    return group_score_df

def format_df(df):
    formatted = df.melt(id_vars=['n_clusters', 'hemi'], 
                        var_name='metric', 
                        value_name='score')
    formatted = formatted[formatted['metric'] == 'silhouette_score']
    formatted = formatted[formatted['n_clusters'] < 6 ]
    return formatted

def plot_distri(df, ax) : 

    sns.violinplot(data=df, x='n_clusters', y='score', 
                hue='hemi', ax=ax, alpha=0.7)
    sns.stripplot(data=df, x='n_clusters', y='score', 
                hue='hemi', ax=ax, dodge=True, alpha=0.5, jitter=0.4)
    ax1.set_title('Score Distribution by Number of Clusters')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')

def plot_median(df, ax) : 
    sns.lineplot(data=df, x='n_clusters', y='score', 
                    hue='hemi', err_style='band', ax=ax, estimator='median')
    ax.set_title('Score Trends')

def plot_median_heat(df, ax) : 
    pivot_scores = df.pivot_table(
        values='score', 
        index='hemi',
        columns='n_clusters',
        aggfunc='median'
    )
    sns.heatmap(pivot_scores, annot=True, cmap='RdYlBu_r', 
                center=pivot_scores.mean().mean(), ax=ax)
    ax.set_title('Median Scores by Clusters and Hemisphere')

def plot_stability(df, ax) : 
    cv_scores = df.pivot_table(
        values='score',
        index='hemi',
        columns='n_clusters',
        aggfunc=lambda x: x.mean()/x.std()
    )
    sns.heatmap(cv_scores, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Score Stability (CV)')

def plot_optimality(df, ax) : 
    # Find optimal number of clusters
    # Combine median score and stability
    medians = df.pivot_table(values='score', 
                                    index='hemi', 
                                    columns='n_clusters', 
                                    aggfunc='median')
    stability = df.pivot_table(values='score', 
                                        index='hemi',
                                        columns='n_clusters',
                                        aggfunc=lambda x : x.mean()/x.std())

    # Score that balances high median with low CV
    optimality = medians * stability
    sns.heatmap(optimality, annot=True, cmap='viridis', ax=ax)
    ax.set_title('Optimal Number of Clusters')



# Main
if __name__ == "__main__" :
    parcel_output_dir = "results/parcel_output"

    subject_ids, subject_files = get_subject_ids(parcel_output_dir)
    group_score_df = get_group_score_df(parcel_output_dir)
    group_score_df.to_csv("results/group_analysis/group_score-silhouette.csv", index=False)

    df = format_df(group_score_df)

    fig = plt.figure(figsize=(15, 10))
    # Premier rangée de plots (2 graphiques)
    ax1 = plt.subplot(3, 2, 1)
    plot_distri(df, ax1)

    ax2 = plt.subplot(3, 2, 2)
    plot_median(df, ax2)

    # Deuxième rangée de plots (2 graphiques)
    ax3 = plt.subplot(3, 2, 3)
    plot_median_heat(df, ax3)

    ax4 = plt.subplot(3, 2, 4)
    plot_stability(df, ax4)

    # Troisième rangée de plots (1 graphique sur toute la largeur)
    ax5 = plt.subplot(3, 1, 3)
    plot_optimality(df, ax5)

    plt.tight_layout()
    plt.savefig("results/figures/optimal_score_analysis.pdf")

    plt.show()


# Run analysis
