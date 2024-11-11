# Brain Parcellation Project

A Python-based neuroimaging analysis pipeline for functional parcellation of brain regions using connectivity-based clustering.

## Overview

This project implements a complete pipeline for:
- Functional parcellation of brain regions (focusing on Frontal Operculum)
- Connectivity analysis with HCP-MMP1 atlas
- Morphometry analysis
- Group-level statistics and visualization

## Installation

### Requirements
- Python 3.9.6
- Required packages:

```bash
nibabel==3.2.1
numpy==1.23.5
nilearn==0.10.4
matplotlib==3.6.2
pandas==1.3.3
scipy
scikit-learn

.
├── data/
│   ├── anat/          # Anatomical images
│   ├── func/          # Functional MRI data
│   ├── mask_ROI/      # ROI masks
│   ├── atlas/         # HCP-MMP1 atlas files
│   └── segmentation/  # Brain segmentation files
├── src/
│   ├── parcel_core/   # Core functionality
│   │   ├── connectivity.py
│   │   └── clustering.py
│   ├── utils/         # Utility functions
│   ├── main_parcel.py # Main pipeline script
│   └── relabelling.py # Group-level relabeling
├── results/           # Output directory
└── logs/             # Processing logs

```

## Usage

1. Data Preparation
Format your data according to BIDS specification:
```
python src/format_data.py
```

2. Individual Analysis
Run parcellation pipeline for all subjects:
```
python src/main_parcel.py
```

#### This performs:

- ROI time series extraction
- Spectral clustering
- Whole-brain connectivity
- Atlas-based connectivity

3. Group Analysis

#### Extract morphometry
```
python src/extract_morphometry.py
```
#### Perform cluster relabeling
```
python src/relabelling.py
```
#### Generate connectivity visualizations
```bash
python src/spider_connectivity.py

results/
├── parcel_output/              # Individual results
│   └── sub-XXX/
│       ├── cluster/           # Cluster masks
│       ├── connectivity/      # Connectivity maps
│       ├── figures/          # Visualizations
│       └── timeseries/       # Extracted timecourses
├── group_analysis/            # Group-level results
│   ├── morphometry/          # ROI morphometry
│   └── relabelling/         # Cluster correspondence
└── figures/                  # Group visualizations
```

## Key Features

### 1. Parcellation
- Spectral clustering of functional connectivity
- Automatic optimal cluster number detection 
- Individual-level quality control

### 2. Connectivity Analysis
- Whole-brain connectivity maps
- Atlas-based regional connectivity
- Statistical comparison between hemispheres

### 3. Morphometry
- Volume calculations
- Center of mass coordinates
- Shape metrics

### 4. Visualization
- Surface projections
- Spider plots for connectivity
- 3D cluster visualizations

## References
- HCP-MMP1 Atlas: Glasser et al. (2016)
- Spectral Clustering: von Luxburg (2007)
- Brain Surface Visualization: Nilearn

## Authors
- Charles Verstraete (<charlesverstraete@outlook.com>)