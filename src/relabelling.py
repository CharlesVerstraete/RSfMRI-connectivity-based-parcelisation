# python 3.9.6
# -*- coding: utf-8 -*-
"""
relabelling.py
=========

Description:
    Relabelling clusters with similar anatomic position and connectivity profile

Author:
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""

# Import section
import numpy as np
import os
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Function definitions
def get_subject_map(base_dir : str, subject_id: str, hemisphere: str) -> dict:
    """Load all relevant data for a subject's parcellation"""
    subject_dir = os.path.join(base_dir, f'sub-{subject_id}')
    
    return {
        # Load cluster masks
        'clusters': [
            nib.load(os.path.join(subject_dir, 'cluster', f'sub-{subject_id}_hemi-{hemisphere}_parcel-{i}.nii')).get_fdata()
            for i in range(1, 4)  # Assuming 3 clusters
        ],
        # Load connectivity maps
        'connectivity': [
            nib.load(os.path.join(subject_dir, 'connectivity', f'sub{subject_id}_hemi-{hemisphere}_parcel-{i}_connectivity-map.nii')).get_fdata()
            for i in range(1, 4)
        ],
        # Load atlas connectivity
        'atlas_conn': pd.read_csv(os.path.join(subject_dir, 'connectivity', f'sub{subject_id}_HCP-MMP1-{hemisphere}_connectivity.csv'))
    }

def get_centers(subjects_map: dict) -> np.ndarray:
    """Get centers of mass for each cluster in each subject"""
    centers = np.zeros((len(subjects_map), 3, 3))
    for idx, (_, data) in enumerate(subjects_map.items()):
        centers[idx] = np.array([
            np.mean(np.where(cluster), axis=1) 
            for cluster in data['clusters']
        ])
    return centers

def create_group_map(base_dir : str, hemisphere: str) -> dict:
    """Create group map for all subjects"""

    subject_files = os.listdir(base_dir)
    if '.DS_Store' in subject_files:
        subject_files.remove('.DS_Store')
    subject_files.sort()
    subject_ids = [x.split('-')[-1] for x in subject_files]

    subjects_map = {}
    for subject in subject_ids:
        subjects_map[subject] = get_subject_map(base_dir, subject, hemisphere)

    return subject_files, subject_ids, subjects_map

def get_centers_projecttion_limits(centers: np.ndarray) -> tuple:
    """Get limits for projection of cluster centers"""
    average_x = np.mean(centers[:, :, 0])
    average_y = np.mean(centers[:, :, 1])

    max_x = np.max(centers[:, :, 0])+1
    min_x = np.min(centers[:, :, 0])-1
    max_y = np.max(centers[:, :, 1])+1
    min_y = np.min(centers[:, :, 1])-1

    return average_x, average_y, max_x, min_x, max_y, min_y

def plot_individual_projection(
        centers: np.ndarray, 
        subject_ids: list, 
        colors: list = ['firebrick', 'royalblue', 'forestgreen'], 
        save_path : str = None, 
        hemi : str = "") -> None:
    
    """Plot projection of cluster centers"""

    average_x, average_y, max_x, min_x, max_y, min_y = get_centers_projecttion_limits(centers)

    nrow = np.sqrt(len(subject_ids)).astype(int)
    ncol = np.ceil(len(subject_ids) / nrow).astype(int)

    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 12), sharex=True, sharey=True)
    plt.subplots_adjust(hspace=0.3, wspace=0.1, top=0.95, bottom=0.05, left=0.05, right=0.95)

    for idx in range(len(subject_ids)):
        ax = axes.flat[idx]
        for i in range(3):
            ax.scatter(
                centers[idx, i, 0],  # X coordinate
                centers[idx, i, 1],  # Z coordinate
                c=colors[i],
                label=f'Cluster {i+1}',
                alpha=0.8
        )
            ax.axhline(average_y, color='black', linestyle='--', lw = 0.8, alpha=0.5, label='Average y')
            ax.axvline(average_x, color='black', linestyle='--', lw = 0.8, alpha=0.5, label='Average x')

            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_title(f'Subject {subject_ids[idx]}')

    if len(subject_ids) < nrow * ncol:
        for idx in range(len(subject_ids), nrow*ncol):
            axes.flat[idx].axis('off')

    legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', 
                markerfacecolor=color, label=f'Cluster {i+1}', markersize=8)
    for i, color in enumerate(colors)
    ] + [
        plt.Line2D([0], [0], color='black', linestyle='--', label='Average')
    ]
    
    fig.legend(handles=legend_elements, loc='lower right', bbox_to_anchor=(0.95, 0.1))
    if save_path:
        plt.savefig(os.path.join(save_path, f'group-hemi-{hemi}-projection_individual.pdf'))
    plt.show()


def plot_3d_centers(centers: np.ndarray, subject_ids: list, relabelled_clusters, colors: list = ['firebrick', 'royalblue', 'forestgreen'], save_path = None, hemi = "") -> None:
    """Plot 3D projection of cluster centers"""

    z_avg = np.mean(centers[:, :, 2])
    _, _, x_max, x_min, y_max, y_min = get_centers_projecttion_limits(centers)

    fig = plt.figure(figsize=(15, 7))
    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    # Add Z-plane

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10),
                        np.linspace(y_min, y_max, 10))
    zz = np.full_like(xx, z_avg)
    ax1.plot_surface(xx, yy, zz, alpha=0.3, color='dimgray')
    # Plot clusters
    for idx in range(len(subject_ids)):
        relabel = relabelled_clusters.loc[idx].values[-3:]
        for i in range(3):
            ax1.scatter(
                centers[idx, relabel[i]-1, 0],  
                centers[idx, relabel[i]-1, 1],
                centers[idx, relabel[i]-1, 2],  
                c=colors[i],
                alpha=0.8,
                s=100
            )

    ax1.set_xlabel('X (Lateral-Medial)')
    ax1.set_ylabel('Y (Anterior-Posterior)') 
    ax1.set_zlabel('Z (Superior-Inferior)')
    ax1.set_title(f'3D View (Z-avg = {z_avg:.1f})')
    ax1.set_box_aspect([1,1,1])
    ax1.view_init(elev=10, azim=-10, roll = 0)

    # 2D projection
    ax2 = fig.add_subplot(122)
    for idx in range(len(subject_ids)):
        relabel = relabelled_clusters.loc[idx].values[-3:]
        for i in range(3):
            ax2.scatter(
                centers[idx, relabel[i]-1, 0],  
                centers[idx, relabel[i]-1, 1],
                c=colors[i],
                label=f'Cluster {i+1}' if idx == 0 else None,
                alpha=0.8
            )
    ax2.set_xlabel('X (Lateral-Medial)')
    ax2.set_ylabel('Y (Anterior-Posterior)')
    ax2.set_title('Sagittal View')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.legend(bbox_to_anchor=(1.0, 0.5))

    if save_path:
        plt.savefig(os.path.join(save_path, f'group-hemi-{hemi}-3d_projection.pdf'))
    plt.show()

def extract_connectivity(subjects_map: dict, relabelled_clusters : pd.DataFrame) -> np.ndarray:
    """ Extract connectivity matrix for all subjects"""
    connectivity = np.zeros((len(subjects_map), 3, 592895))
    for idx, data in enumerate(subjects_map.values()):
        for i in range(len(data['connectivity'])):
            connectivity[idx, i, :] += data['connectivity'][relabelled_clusters.iloc[idx, i+1]-1].flatten()
    return connectivity

def plot_similarity(connectivity: np.ndarray, save_path : str = None, hemi : str = "") -> None:
    """ Plot similarity matrix"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Connectivity z-Fisher Matrices', fontsize=14, y=0.95)

    # Labels for rows and columns
    cluster_labels = ['Anterior', 'Middle', 'Posterior']

    # Calculate similarities and plot
    for j in range(3):
        for i in range(3):
            similarity = np.arctanh(np.clip(np.corrcoef(connectivity[:, i, :], connectivity[:, j, :])[:31, 31:], -0.999, 0.999))
            im = axes[i, j].imshow(similarity, 
                                cmap='coolwarm', 
                                aspect='auto', 
                                origin='lower',
                                vmin=-1, 
                                vmax=1)
            
            # Add titles
            if i == 0:
                axes[i, j].set_title(f'Cluster {j+1}\n({cluster_labels[j]})')
            if j == 0:
                axes[i, j].set_ylabel(f'Cluster {i+1}\n({cluster_labels[i]})')

    # Add single colorbar
    cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('z', rotation=270, labelpad=15)

    # Adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, right=0.9)
    if save_path:
        plt.savefig(os.path.join(save_path, f'group-hemi-{hemi}-inter_subject_parcels_correlation.pdf'))
    plt.show()


if __name__ == "__main__":

    # Main section
    base_dir = "results/parcel_output"
    output_relabel_dir = "results/group_analysis/relabelling"
    output_figures_dir = "results/figures"
    colors = ['firebrick', 'royalblue', 'forestgreen']

    # Left hemisphere
    subject_files, subject_ids, subjects_map = create_group_map(base_dir, 'left')
    centers = get_centers(subjects_map)
    plot_individual_projection(centers, subject_ids, save_path=output_figures_dir, hemi='left')

    relabelled_clusters = pd.read_csv(os.path.join(output_relabel_dir, "relabelling_cluster_left.csv"), sep=';')
    plot_3d_centers(centers, subject_ids, relabelled_clusters, save_path=output_figures_dir, hemi='left')

    connectivity = extract_connectivity(subjects_map, relabelled_clusters)
    plot_similarity(connectivity, save_path=output_figures_dir, hemi='left')

    # Right hemisphere
    subject_files, subject_ids, subjects_map = create_group_map(base_dir, 'right')
    centers = get_centers(subjects_map)
    plot_individual_projection(centers, subject_ids, save_path=output_figures_dir, hemi='right')

    relabelled_clusters = pd.read_csv(os.path.join(output_relabel_dir, "relabelling_cluster_right.csv"), sep=';')
    plot_3d_centers(centers, subject_ids, relabelled_clusters, save_path=output_figures_dir, hemi='right')

    connectivity = extract_connectivity(subjects_map, relabelled_clusters)
    plot_similarity(connectivity, save_path=output_figures_dir, hemi='right')










