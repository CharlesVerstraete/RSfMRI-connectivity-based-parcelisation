# python 3.9.6
# -*- coding: utf-8 -*-
"""
extract_morphometry.py
======================

Description:
    Extract morphometry data from T1w MRI, mask of ROI and the segmentation

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

import os
import sys
import nibabel as nib
import shutil
import pandas as pd
import numpy as np
from utils.morphometry_tools import *


if __name__ == "__main__":
    # Create results directory
    roi_dir = "data/mask_ROI"
    output_dir = "results/morphometry"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize results dataframe
    results = []

    # Process each subject
    for filename in sorted(os.listdir(roi_dir)):
        # Check if file matches mask naming pattern
        if not filename.endswith('_mask.nii'):
            continue
            
        mask_path = os.path.join(roi_dir, filename)
        
        # Extract subject ID and hemisphere from filename
        parts = filename.split('_')
        subject = parts[0]  # sub-XXX
        hemi = parts[1].split('-')[1]  # L or R from hemi-L
        
        # Extract morphometry
        metrics = extract_morphometry(mask_path)
        if metrics:
            metrics.update({
                'subject': subject,
                'hemisphere': hemi
            })
            results.append(metrics)
            print(f"Processed: {filename}")
        else:
            print(f"Failed to process: {filename}")

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'morphometry.csv'), index=False)
    print(f"Saved morphometry results for {len(results)} masks")



