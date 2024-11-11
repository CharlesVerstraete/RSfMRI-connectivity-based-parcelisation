# python 3.9.6
# -*- coding: utf-8 -*-
"""
morphometry_tools.py
======================

Description:
    Morphometry tools for extracting morphometry data from T1w MRI, mask of ROI and the segmentation    

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
import nibabel as nib
import pandas as pd
import numpy as np
from src.utils.format_helper import *

def extract_voxel_features(voxels: np.ndarray) -> tuple:
    """Extract morphometric features from voxel coordinates."""
    min_coords = voxels.min(axis=0)
    max_coords = voxels.max(axis=0)
    com = voxels.mean(axis=0)
    return min_coords, max_coords, com

def vox_to_mni(vox: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Convert voxel coordinates to MNI space."""
    return nib.affines.apply_affine(affine, vox).astype(int)

def get_volume(voxels: np.ndarray, voxel_dims: np.ndarray) -> float:
    """Calculate volume from voxel coordinates."""
    voxel_size = np.prod(voxel_dims)
    return len(voxels) * voxel_size


def extract_morphometry(mask_path: str) -> dict:
    img = nib.load(mask_path)
    data = img.get_fdata()
    affine = img.affine

    # Get mask variables
    voxels = np.array(np.where(data > 0)).T
    n_voxels = len(voxels)
    voxel_dims = img.header.get_zooms()

    # Voxel features
    min_coords, max_coords, com = extract_voxel_features(voxels)
    min_mni, max_mni, com_mni = vox_to_mni(min_coords, affine), vox_to_mni(max_coords, affine), vox_to_mni(com, affine)

    # Volume calculations
    volume = get_volume(voxels, voxel_dims)

    # Dimensions
    dimensions = max_coords - min_coords
    dimensions_mm = dimensions * voxel_dims

    return {
        # Existing metrics
        'volume_mm3': volume,
        'n_voxels': n_voxels,
        'x_min': min_coords[0],
        'y_min': min_coords[1],
        'z_min': min_coords[2],
        'x_max': max_coords[0],
        'y_max': max_coords[1],
        'z_max': max_coords[2],
        'com_x': com[0],
        'com_y': com[1],
        'com_z': com[2],
        'x_min_mni': min_mni[0],
        'y_min_mni': min_mni[1],
        'z_min_mni': min_mni[2],
        'x_max_mni': max_mni[0],
        'y_max_mni': max_mni[1],
        'z_max_mni': max_mni[2],
        'com_x_mni': com_mni[0],
        'com_y_mni': com_mni[1],
        'com_z_mni': com_mni[2],
        'length': dimensions[0],
        'width': dimensions[1],
        'height': dimensions[2],
        'length_mm': dimensions_mm[0],
        'width_mm': dimensions_mm[1],
        'height_mm': dimensions_mm[2],
    }


def groupaverage_mask_from_dir(input_dir : str, fileslist: list,  output_path: str) -> None:
    """Create group average mask from list of masks."""
    mask_paths = [os.path.join(input_dir, f) for f in fileslist]
    first_mask = nib.load(mask_paths[0])
    shape = first_mask.shape
    affine = first_mask.affine
    mask_data = np.zeros(shape)
    for mask_path in mask_paths:
        mask_data += nib.load(mask_path).get_fdata()
    mask_data = mask_data / len(mask_paths)
    save_nifti(mask_data, affine, output_path)
    print(f"Group average mask saved to {output_path}")