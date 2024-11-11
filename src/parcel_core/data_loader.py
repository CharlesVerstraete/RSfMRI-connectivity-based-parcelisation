import nibabel as nib

class ImageLoader:
    """Handles loading and basic preprocessing of neuroimaging data."""
    def __init__(self, anat_path: str, func_path: str):
        self.anat_img = nib.load(anat_path)
        self.func_img = nib.load(func_path)
        self.anat_data = self.anat_img.get_fdata()
        self.func_data = self.func_img.get_fdata()
        self.anat_affine = self.anat_img.affine
        self.func_affine = self.func_img.affine