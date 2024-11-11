from nilearn import datasets, surface, plotting
from scipy.spatial import KDTree
import numpy as np


def surface_resample(surf_to_resample, orig_coord, target_coord):
    tree = KDTree(orig_coord)
    _, indices = tree.query(target_coord)
    return surf_to_resample[indices]


# lh_atlas_src = "data/atlas/lh.HCP-MMP1.annot"
# lh_atlas = surface.load_surf_data(lh_atlas_src)

# fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
# fsaverage5 = datasets.fetch_surf_fsaverage("fsaverage3")

# coords_fsaverage, _ = surface.load_surf_data(fsaverage["pial_left"])
# coords_fsaverage5, _ = surface.load_surf_mesh(fsaverage5["pial_left"])

# lh_atlas_fsaverage5 = surface_resample(lh_atlas, coords_fsaverage, coords_fsaverage5)

# lh_atlas_fsaverage5_filtered = np.zeros(lh_atlas_fsaverage5.shape)



# lh_atlas_src = "data/atlas/lh.HCP-MMP1.annot"
# lh_atlas = surface.load_surf_data(lh_atlas_src)
# lh_atlas_filtered = np.zeros(lh_atlas.shape)

# for i, j in enumerate([80, 31, 43, 77, 100, 124, 156]) :
#     lh_atlas_filtered[lh_atlas == j] = i

# plotting.plot_surf_roi(
#     fsaverage5['pial_left'],
#     roi_map=lh_atlas_filtered, 
#     hemi="left", 
#     bg_map=fsaverage5['sulc_left'],
#     view="lateral", 
#     cmap="hsv", 
#     bg_on_data=True)



# lh_atlas_fsaverage5_filtered[lh_atlas_fsaverage5 == j] = i
# lh_atlas_fsaverage_filtered = np.zeros(lh_atlas.shape)

# lh_atlas_fsaverage_filtered[lh_atlas == 1] = 1


# plotting.plot_surf_contours(
#     fsaverage['pial_left'], roi_map=lh_atlas_fsaverage_filtered, hemi='left', view='lateral', bg_map=fsaverage['sulc_left'], cmap='hsv', bg_on_data=True
# )
# plotting.show()


# # Visualiser le résultat dans fsaverage5
# plotting.plot_surf_roi(fsaverage5['pial_left'], roi_map=lh_atlas_fsaverage5_filtered, hemi="left", bg_map=fsaverage5['sulc_left'],
#                        view="lateral", cmap="hsv", bg_on_data=True)

# plotting.plot_surf_roi(fsaverage['pial_left'], roi_map=lh_atlas, hemi="left", view="lateral", bg_map=fsaverage['sulc_left'], cmap="hsv", alpha=0.3)
# plotting.show()
