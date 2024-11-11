import nibabel as nib
from nilearn.image import resample_img
import nilearn.masking as nimsk
import numpy as np

class ROIProcessor:
    """Handles ROI mask processing and resampling."""
    def __init__(self, roi_path: str, target_img: nib.Nifti1Image):
        self.img = nib.load(roi_path)
        self.img.set_data_dtype(np.uint8)
        self.data = self.img.get_fdata()
        self.resampled_img = self.resample_mask(target_img)
        self.resampled_img.set_data_dtype(np.uint8)
        self.resampled_data = self.resampled_img.get_fdata()
        
    def resample_mask(self, target_img : nib.Nifti1Image) -> nib.Nifti1Image:
        '''Resample the mask to the target image.'''
        return resample_img(
            self.img,
            target_affine=target_img.affine,
            target_shape=target_img.shape[:3],
            interpolation='nearest'
        )
    
    def extract_time_series(self, func_data: np.ndarray) -> np.ndarray:
        '''Extract time series data from functional data using the mask.'''
        return nimsk.apply_mask(func_data, self.resampled_img)