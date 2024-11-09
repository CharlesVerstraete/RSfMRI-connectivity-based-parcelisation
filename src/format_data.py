# python 3.9.6
# -*- coding: utf-8 -*-
"""
format_data.py
=========

Description:
    Import data from a directory and format it for BIDS compatibility and analysis.

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
from utils.format_helper import *


if __name__ == "__main__" :
    # Global variables
    subject_fo_data_dir = "/Users/charles.verstraete/Documents/w0_OpFrontal/ParcelProject/subjects/FO"
    n_subjects = len(os.listdir(subject_fo_data_dir))

    # Anatomical 
    bids_anat_dir = os.path.join("data", "anat")
    original_name = f's{i}_anat.nii'
    new_name = "T1w.nii"
    format_loop(subject_fo_data_dir, bids_anat_dir, n_subjects, original_name, new_name, operation="copy")

    # Functional
    bids_func_dir = os.path.join("data", "func")
    original_name = f's{i}_func.nii'
    new_name = "task-rest_bold.nii"
    format_loop(subject_fo_data_dir, bids_func_dir, n_subjects, original_name, new_name, operation="copy")

    # ROI
    df_new_y = pd.read_csv("new_y.csv")
    bids_roi_dir = os.path.join("data", "mask_ROI")

    ## Left hemisphere
    format_loop(
        subject_fo_data_dir, 
        bids_roi_dir,
        n_subjects,
        "maskFOleft_anat.nii",
        "hemi-L_FO_mask.nii",
        operation="cut",
        cuts=df_new_y,
        side="left"
    )    
    ## Right hemisphere
    format_loop(
        subject_fo_data_dir, 
        bids_roi_dir,
        n_subjects,
        "maskFOright_anat.nii",
        "hemi-R_FO_mask.nii",
        operation="cut",
        cuts=df_new_y,
        side="right"
    )
    
    # Segmentation
    bids_segmentation_dir = os.path.join("data", "segmentation")
    original_name = "maskGM_anat.nii"
    new_name = "space-orig_dseg.nii"
    format_loop(subject_fo_data_dir, bids_segmentation_dir, n_subjects, original_name, new_name, operation="binarize")


