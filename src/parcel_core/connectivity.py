import numpy as np
import os
from joblib import Parallel, delayed
import nibabel as nib
from nilearn.masking import unmask
from nilearn import plotting, surface, datasets
import matplotlib.pyplot as plt
from utils.plot_tools import surface_resample

global EPSILON 
EPSILON = 1e-6

class ConnectivityAnalyzer:
    def __init__(self, subject_id, hemi, seg_time_series, averaged_timecourses, seg_mask):
        self.subject = subject_id
        self.hemi = hemi
        self.seg_time_series = seg_time_series
        self.averaged_timecourses = averaged_timecourses
        self.seg_mask = seg_mask
        self.connectivity_matrix = np.zeros((averaged_timecourses.shape[0], seg_time_series.shape[0]))

    # Core correlation methods
    def _compute_correlation(self, timecourse1, timecourse2):
        """Compute correlation between two timecourses with error handling."""
        if np.std(timecourse1) == 0 or np.std(timecourse2) == 0:
            return 0
        corr = np.corrcoef(timecourse1, timecourse2)[0,1]
        return np.arctanh(np.clip(corr, -1 + EPSILON, 1 - EPSILON)) if np.isfinite(corr) else 0
    
    def _process_chunk(self, chunk_start, chunk_size, sub_roi):
        """Process a chunk of time series."""
        chunk_results = np.zeros(chunk_size)
        for i in range(chunk_size):
            idx = chunk_start + i
            if idx < self.seg_time_series.shape[0]:
                chunk_results[i] = self._compute_correlation(
                    self.averaged_timecourses[sub_roi],
                    self.seg_time_series[idx,:]
                )
        return chunk_results

    def _parallel_compute(self, sub_roi, n_jobs):
        """Run parallel computation of correlations."""
        n_voxels = self.seg_time_series.shape[0]
        chunk_size = n_voxels // (os.cpu_count() if n_jobs == -1 else n_jobs)
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._process_chunk)(i, chunk_size, sub_roi)
            for i in range(0, n_voxels, chunk_size)
        )
        return np.concatenate(results)[:n_voxels]

    def plot_connectivity_surface(self, connectivity_map, output_file=None):
        """Plot connectivity on brain surface."""
        plotting.plot_img_on_surf(
            connectivity_map,
            views=["lateral", "medial"],
            hemispheres=["left", "right"],
            colorbar=True,
            cmap="cold_hot",
            threshold=0.2,
            title="Connectivity Map",
            bg_on_data=True,
            symmetric_cbar=True,
            output_file=output_file
        )

    def get_atlas_surface(self, fsaverage_name):
        """Plot HCP-MMP1 atlas on brain surface."""
        hemi_map = {'left': 'lh', 'right': 'rh'}
        atlas = surface.load_surf_data(f"data/atlas/{hemi_map[self.hemi]}.HCP-MMP1.annot")
        
        if fsaverage_name == "fsaverage":
            fsaverage = datasets.fetch_surf_fsaverage(fsaverage_name)
            resampled_atlas = atlas
        else:
            fsaverage_orig = datasets.fetch_surf_fsaverage("fsaverage")
            fsaverage = datasets.fetch_surf_fsaverage(fsaverage_name)
            coords_orig, _ = surface.load_surf_data(fsaverage_orig[f"pial_{self.hemi}"])
            coords_target, _ = surface.load_surf_mesh(fsaverage[f"pial_{self.hemi}"])
            resampled_atlas = surface_resample(atlas, coords_orig, coords_target)

        return resampled_atlas, fsaverage

    def plot_atlas_connectivity(self, views, fsaverage_name, output_dir):
        """Plot atlas connectivity results."""

        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(17, 12), 
                                subplot_kw={'projection': '3d'}, 
                                gridspec_kw={'wspace': 0.1, 'hspace': 0.1, 'left': 0.1, 'right': 0.9, 'top': 0.9, 'bottom': 0.1})

        atlas, fsaverage = self.get_atlas_surface(fsaverage_name)
        # In the plotting loop
        for i in range(3):
            for j, view in enumerate(views):

                top_ids = self.connectivity_df[self.connectivity_df[f"connectivity_subroi{i+1}"] > 0.5]

                atlas_filtered = np.zeros(atlas.shape)
                for _, row in top_ids.iterrows():
                    atlas_filtered[atlas == row['ROI_n']] = row[f'connectivity_subroi{i+1}']
                
                vmin = top_ids[f'connectivity_subroi{i+1}'].min()
                vmax = top_ids[f'connectivity_subroi{i+1}'].max()
                display = plotting.plot_surf_roi(
                    fsaverage[f'pial_{self.hemi}'],
                    roi_map=atlas_filtered,
                    view=view,
                    hemi=f'{self.hemi}',
                    bg_map=fsaverage[f'sulc_{self.hemi}'],
                    cmap='viridis',
                    vmin=vmin,
                    vmax=vmax,
                    bg_on_data=True,
                    axes=axes[i, j],
                    title=f'Sub-ROI {i+1} - {view}',
                    colorbar=True,
                    cbar_tick_format='%.2f'
                )
            top_regions = top_ids['ROI_glasser_2'].unique()
            axes[i, 0].text2D(-0.25, 0.95, 
                f"Top regions:\n" + "\n".join(top_regions),
                transform=axes[i, 0].transAxes,
                fontsize=6,
                verticalalignment='top')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'figures', f'sub{self.subject:03d}_HCP-MMP1-{self.hemi}_connectivity.pdf'))


    # Atlas connectivity methods
    def compute_atlas_correlations(self, atlas_time_series):
        """Compute correlations with atlas regions."""
        n_clusters = self.averaged_timecourses.shape[0]
        corr_mat = np.corrcoef(self.averaged_timecourses, atlas_time_series.T)
        return corr_mat[:n_clusters, n_clusters:]

    def create_atlas_connectivity_df(self, atlas_df, corr_mat):
        """Create DataFrame with atlas connectivity results."""
        self.connectivity_df = atlas_df.copy()
        for i in range(corr_mat.shape[0]):
            self.connectivity_df[f"connectivity_subroi{i+1}"] = corr_mat[i, :]

    # Main interface methods
    def compute_wholebrain_connectivity(self, sub_roi, output_dir=None, n_jobs=-1):
        """Compute whole-brain connectivity for a sub-ROI."""
        self.connectivity_matrix[sub_roi] = self._parallel_compute(sub_roi, n_jobs)
        
        if output_dir:
            connectivity_map = unmask(self.connectivity_matrix[sub_roi], self.seg_mask)
            nib.save(connectivity_map,
                os.path.join(output_dir, 'connectivity', f'sub{self.subject:03d}_hemi-{self.hemi}_parcel-{sub_roi + 1}_connectivity-map.nii'))
            self.plot_connectivity_surface(connectivity_map, 
                os.path.join(output_dir, 'figures', f'sub{self.subject:03d}_hemi-{self.hemi}_parcel-{sub_roi + 1}_connectivity-map.pdf'))

    def compute_atlas_connectivity(self, atlas_time_series, atlas_df, output_dir=None):
        """Compute connectivity with atlas regions."""
        corr_mat = self.compute_atlas_correlations(atlas_time_series)
        self.create_atlas_connectivity_df(atlas_df, corr_mat)
        
        if output_dir:
            out_path = os.path.join(output_dir, 'connectivity', f'sub{self.subject:03d}_HCP-MMP1-{self.hemi}_connectivity.csv')
            self.connectivity_df.to_csv(out_path)
            self.plot_atlas_connectivity(['lateral', 'medial'], "fsaverage5", output_dir)
            



