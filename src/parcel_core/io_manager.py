import os
import nibabel as nib
import numpy as np

class IOManager:
    """Handles file I/O operations."""
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def save_nifti(self, data: np.ndarray, affine: np.ndarray, filename: str):
        img = nib.Nifti1Image(data, affine)
        nib.save(img, os.path.join(self.output_dir, filename))
