# python 3.9.6
# -*- coding: utf-8 -*-
"""
format_helper.py
=========

Description:
    Helper functions for format_data.py

Dependencies:
    - nilearn 0.10.4
    - numpy 1.23.5
    - matplotlib 3.6.2
    
Author:
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""


# Import section
import os
import sys
import nibabel as nib
import shutil
import pandas as pd
import numpy as np

# Function definitions

def copy_original_file(src_path : str, dest_path : str) -> None :
    """ Copy original file to destination """

    shutil.copy2(src_path, dest_path)
    print(f"Successfully copied: {os.path.basename(src_path)} -> {os.path.basename(dest_path)}")


def convert_mni_to_voxel(mni_coord : np.ndarray, affine : np.ndarray) -> np.ndarray :
    """ Convert MNI coordinates to voxel coordinates """

    voxel_coord = np.linalg.inv(affine).dot(mni_coord)
    return voxel_coord.astype(int)


def save_nifti(data : np.ndarray, affine : np.ndarray, dest_path : str) -> None :
    """ Create and save Nifti image """

    new_img = nib.Nifti1Image(data, affine)
    nib.save(new_img, dest_path)
    print(f"Successfully saved: {os.path.basename(dest_path)}")


def cut_posterior_part(src_path : str, dest_path : str, y : int) -> None : 
    """ Cut posterior part of Nifti image """

    img = nib.load(src_path)
    data = img.get_fdata()
    affine = img.affine
    mni_coord = np.array([0, y, 0, 1])
    y_voxel = convert_mni_to_voxel(mni_coord, affine)[1]
    data[:, :y_voxel, :] = 0  # Cut posterior part
    save_nifti(data, affine, dest_path)
    print(f"Successfully cut posterior part of: {os.path.basename(src_path)} -> {os.path.basename(dest_path)} at y = {y}")


def try_copy(src_path : str, dest_path : str) -> list :
    """ Try to copy file """

    counter = [0, 0]
    try:
        if os.path.isfile(src_path):
            copy_original_file(src_path, dest_path)
            counter[0] = 1
        else:
            print(f"WARNING: Functional file not found - {src_path}")
            counter[1] = 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        counter[1] = 1
    return np.array(counter)


def try_cut(src_path :str, dest_path : str, y : int) -> list :
    """ Try to cut posterior part of file """

    counter = [0, 0]
    try:
        if os.path.isfile(src_path):
            cut_posterior_part(src_path, dest_path, y)
            counter[0] = 1
        else:
            print(f"WARNING: Source file not found - {src_path}")
            counter[1] = 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        counter[1] = 1
    return np.array(counter)


def try_binarize(src_path : str, dest_path : str) -> list :
    """ Try to binarize file """

    counter = [0, 0]
    try:
        if os.path.isfile(src_path):
            img = nib.load(src_path)
            data = img.get_fdata()
            data = np.rint(data).astype(np.uint8)
            data[data != 0] = 1
            save_nifti(data, img.affine, dest_path)
            counter[0] = 1
        else:
            print(f"WARNING: Source file not found - {src_path}")
            counter[1] = 1
    except Exception as e:
        print(f"ERROR: {str(e)}")
        counter[1] = 1
    return np.array(counter)


def format_loop(src_dir : str, 
                dest_dir : str, 
                n_subjects : int,
                original_name : str, 
                new_name : str, 
                operation : str, 
                cuts : pd.DataFrame = None, 
                side :str = None) -> None : 
    """ Loop through subjects """
    counter = np.zeros(2)
    for i in range(1, n_subjects):
        if operation == "copy":
            src_path = os.path.join(src_dir, f"s_{i}",  f"s{i}_{original_name}")
            dest_path = os.path.join(dest_dir, f"sub-{i:03d}_{new_name}")
            counter += try_copy(src_path, dest_path)
        elif operation == "cut":
            src_path = os.path.join(src_dir, f"s_{i}", side,  original_name)
            dest_path = os.path.join(dest_dir, f"sub-{i:03d}_{new_name}")
            y = cuts.loc[i-1, side]
            if y == 0 :
                counter += try_copy(src_path, dest_path)
            else :
                counter += try_cut(src_path, dest_path, y)
        elif operation == "binarize":
            src_path = os.path.join(src_dir, f"s_{i}", f"s{i}_{original_name}")
            dest_path = os.path.join(dest_dir, f"sub-{i:03d}_{new_name}")
            counter += try_binarize(src_path, dest_path)

    print(f"\nSummary:")
    print(f"Successfully processed: {counter[0]}")
    print(f"Failed: {counter[1]}")