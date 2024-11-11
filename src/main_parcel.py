# python 3.9.6
# -*- coding: utf-8 -*-
"""
main_parcel.py
=========

Description:
    Main launcher for the parcel, loops through all subjects and performs the clustering.

Author:
    Charles Verstraete <charlesverstraete@outlook.com>

Created: 
    2024-11

References:
    - Reference papers or documentation

"""

#Import
import os
from parcel import Parcel
import nibabel as nib
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
import os
from time import time
from tqdm import tqdm

# Setup logging with tqdm compatibility
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# Setup logging
def setup_logging(subject_id, hemi):
    # Create logs directory
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Configure file handler
    log_file = os.path.join(log_dir, f'sub-{subject_id:03d}_{hemi}.log')
    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    
    # Configure console handler with tqdm compatibility
    console_handler = TqdmLoggingHandler()
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    
    # Get logger
    logger = logging.getLogger(f'sub-{subject_id:03d}_{hemi}')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Configuration
subjects = range(1, 32)
hemispheres = ['left', 'right']
hemi_map = {'left': 'L', 'right': 'R'}
total_iterations = len(subjects) * len(hemispheres)


atlas_left = nib.load("data/atlas/HCP-MMP1_left.nii")
atlas_right = nib.load("data/atlas/HCP-MMP1_right.nii")
atlas_ref = pd.read_csv("data/atlas/HCP-MMP1_labels.csv", sep=";", header=0)

atlas = {'left': atlas_left, 'right': atlas_right}

# Main processing loop
total_start = time()
with tqdm(total=len(subjects)*len(hemispheres)) as pbar:
    for subject_id in subjects:
        for hemi in hemispheres:
            start_time = time()
            
            # Setup logger for this subject/hemisphere
            logger = setup_logging(subject_id, hemi)
            logger.info(f"Starting processing subject {subject_id:03d} - {hemi}")
            
            try:
                # Load data
                logger.info("Loading data...")
                data_paths = {
                    'anat': f'data/anat/sub-{subject_id:03d}_T1w.nii',
                    'func': f'data/func/sub-{subject_id:03d}_task-rest_bold.nii.gz',
                    'roi': f'data/mask_ROI/sub-{subject_id:03d}_hemi-{hemi_map[hemi]}_FO_mask.nii',
                    'seg': f'data/segmentation/sub-{subject_id:03d}_space-orig_dseg.nii'
                }
                
                # Create output directory
                            # Create directories
                output_dir = f'results/parcel_output/sub-{subject_id:03d}'
                os.makedirs(output_dir, exist_ok=True)

                subdirs = [
                    'cluster',
                    'connectivity',
                    'timeseries',
                    'figures',
                    'metrics',
                    'atlas'
                ]

                for subdir in subdirs:
                    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
                logger.info("Output directories created")


                # Initialize processing
                logger.info("Initializing parcellation...")
                parcel = Parcel(subject_id, hemi, data_paths, output_dir)
                
                # Clustering
                logger.info("Starting clustering...")
                parcel.init_clustering()
                logger.info("Searching optimal clusters...")
                parcel.search_optimal_clusters()
                logger.info("Performing clustering k=3...")
                parcel.perform_clustering(3)
                
                # Connectivity
                logger.info("Computing connectivity...")
                parcel.init_connectivity()
                parcel.perform_wholebrain_connectivity()
                atlas_hemi = atlas[hemi]
                parcel.perform_atlas_connectivity(atlas_hemi, atlas_ref)
                
                # Log completion
                elapsed = time() - start_time
                logger.info(f"Completed in {elapsed:.1f}s")
                
                pbar.update(1)
                pbar.set_postfix({'Subject': f'{subject_id:03d}', 'Hemi': hemi})
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                pbar.update(1)
                continue

logger.info(f"Total time: {time() - total_start:.1f}s")











# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Start timing
# start_time = time()

# subject_id = 1
# hemi = 'left'
# logger.info(f"Processing subject {subject_id:03d}")

# # Load data paths
# data_paths = {
#     'anat' : f'data/anat/sub-{subject_id:03d}_T1w.nii',
#     'func' : f'data/func/sub-{subject_id:03d}_task-rest_bold.nii.gz',
#     'roi' : f'data/mask_ROI/sub-{subject_id:03d}_hemi-L_FO_mask.nii',
#     'seg' : f'data/segmentation/sub-{subject_id:03d}_space-orig_dseg.nii'
# }
# logger.info("Data paths configured")

# # Create directories
# output_dir = f'results/parcel_output/sub-{subject_id:03d}'
# os.makedirs(output_dir, exist_ok=True)

# subdirs = [
#     'cluster',          # Clustering results
#     'connectivity',     # Connectivity maps
#     'timeseries',      # Extracted timeseries
#     'figures',         # All plots
#     'metrics',         # Statistics and metrics
#     'atlas'            # Atlas-related results
# ]

# for subdir in subdirs:
#     os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
# logger.info("Output directories created")

# # Load atlas data
# logger.info("Loading atlas data...")
# atlas_left = nib.load("data/atlas/HCP-MMP1_left.nii")
# atlas_right = nib.load("data/atlas/HCP-MMP1_right.nii")
# atlas_ref = pd.read_csv("data/atlas/HCP-MMP1_labels.csv", sep=";", header=0)
# logger.info("Atlas data loaded successfully")

# # Initialize parcellation
# logger.info("Initializing parcellation...")
# parcel = Parcel(subject_id, hemi, data_paths, output_dir)

# # Run parcellation steps
# logger.info("Starting clustering analysis...")
# parcel.init_clustering()
# logger.info("Searching for optimal number of clusters...")
# parcel.search_optimal_clusters()
# logger.info("Performing final clustering with k=3...")
# parcel.perform_clustering(3)

# # Run connectivity analysis
# logger.info("Starting connectivity analysis...")
# parcel.init_connectivity()
# logger.info("Computing whole brain connectivity...")
# parcel.perform_wholebrain_connectivity()
# logger.info("Computing atlas connectivity...")
# parcel.perform_atlas_connectivity(atlas_left, atlas_ref)

# # Finish
# elapsed_time = time() - start_time
# logger.info(f"Processing completed in {elapsed_time:.1f} seconds")
