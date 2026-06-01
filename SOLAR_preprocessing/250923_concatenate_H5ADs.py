# 250923 Concatenate H5ADs

# Dependencies
# use spt.oct
import scanpy as sc
import squidpy as sq
import pandas as pd
import plotnine as pn
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import anndata as ad

# %%
sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=300, facecolor='white',figsize=(12,12))  # low dpi (dots per inch) yields small inline figures

# %%
seed = 250923

# # Import

# ## EBE00222_S5
# 
# Because this slide has the most samples, plan to work around it

# %%
# ### 74

# %%
adata_74  = ad.read_h5ad("../EBE00222_S5/adata/bottom_center_74_processed.h5ad")
adata_74
adata_74.obs = adata_74.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_74.obs['simplified_ID'] = "14day_5"
adata_74

# %%
theta = np.radians(285) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_74.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_74.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_74.obsm['spatial'] + np.array([3500, 24000])
adata_74.obsm['spatial'] = shifted_coordinates

# %% [markdown]
# ### 75

# %%
adata_75  = ad.read_h5ad("../EBE00222_S5/adata/bottom_left_75_processed.h5ad")
adata_75
adata_75.obs = adata_75.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_75.obs['simplified_ID'] = "14day_6"
adata_75

# %%
theta = np.radians(290) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_75.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_75.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_75.obsm['spatial'] + np.array([-18500, 36000])
adata_75.obsm['spatial'] = shifted_coordinates

# %%
# ### 76

# %%
adata_76  = ad.read_h5ad("../EBE00222_S5/adata/top_left_76_processed.h5ad")
adata_76
adata_76.obs = adata_76.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_76.obs['simplified_ID'] = "14day_7"
adata_76


# %%
theta = np.radians(90) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_76.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_76.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_76.obsm['spatial'] + np.array([16000, 11000])
adata_76.obsm['spatial'] = shifted_coordinates

# %% 
# ### 58

# %%
adata_58  = ad.read_h5ad("../EBE00222_S5/adata/top_center_58_processed.h5ad")
adata_58
adata_58.obs = adata_58.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_58.obs['simplified_ID'] = "14day_2"
adata_58

# %%
shifted_coordinates = adata_58.obsm['spatial'] + np.array([-10000, 0])
adata_58.obsm['spatial'] = shifted_coordinates

theta = np.radians(90) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_58.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_58.obsm['spatial'] = rotated_coords

# %% 
# ### 59

# %%
adata_59  = ad.read_h5ad("../EBE00222_S5/adata/top_right_59_processed.h5ad")
adata_59
adata_59.obs = adata_59.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_59.obs['simplified_ID'] = "14day_3"
adata_59

# %%
theta = np.radians(50) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_59.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_59.obsm['spatial'] = rotated_coords


shifted_coordinates = adata_59.obsm['spatial'] + np.array([10000, -30500])
adata_59.obsm['spatial'] = shifted_coordinates

# %% 
# ### 60

# %%
adata_60  = ad.read_h5ad("../EBE00222_S5/adata/bottom_right_60_processed.h5ad")
adata_60
adata_60.obs = adata_60.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_60.obs['simplified_ID'] = "14day_4"
adata_60

# %%
theta = np.radians(160) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_60.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_60.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_60.obsm['spatial'] + np.array([24000, 24500])
adata_60.obsm['spatial'] = shifted_coordinates

# %% 
# ## EBE00222_S3

# %% 
# ### right

# %%
adata_right  = ad.read_h5ad("../EBE00222_S3/adata/right_processed.h5ad")
adata_right
adata_right.obs = adata_right.obs.rename(columns={'mean_intensity_pi': 'mean_intensity_seg', 'max_intensity_pi': 'max_intensity_seg', 'min_intensity_pi':'min_intensity_seg'})
adata_right.obs['simplified_ID'] = "14day_1"
adata_right

# %%
theta = np.radians(205) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_right.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_right.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_right.obsm['spatial'] + np.array([16000, 7000])
adata_right.obsm['spatial'] = shifted_coordinates

# %% 
# ## EBE00219_S3

# %% 
# ### big

# %%
adata_big  = ad.read_h5ad("../EBE00219_S3/adata/big_processed.h5ad")
adata_big
adata_big.obs = adata_big.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_big.obs['simplified_ID'] = "07day_1"
adata_big


# %%
theta = np.radians(250) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_big.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_big.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_big.obsm['spatial'] + np.array([-1000, -3000])
adata_big.obsm['spatial'] = shifted_coordinates

# %% 
# ### small

# %%
adata_small  = ad.read_h5ad("../EBE00219_S3/adata/small_processed.h5ad")
adata_small
adata_small.obs = adata_small.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_small.obs['simplified_ID'] = "07day_2"
adata_small

# %%
theta = np.radians(174) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_small.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_small.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_small.obsm['spatial'] + np.array([15300, 1200])
adata_small.obsm['spatial'] = shifted_coordinates

# %% 
# ## EBE00216_S4

# %% 
# ### bottom

# %%
adata_bottom  = ad.read_h5ad("../EBE00216_S4/adata/bottom_73_processed.h5ad")
adata_bottom
adata_bottom.obs = adata_bottom.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_bottom.obs['simplified_ID'] = "04day_2"
adata_bottom

# %%
theta = np.radians(40) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_bottom.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_bottom.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_bottom.obsm['spatial'] + np.array([-13000, -6100])
adata_bottom.obsm['spatial'] = shifted_coordinates


# %% 
# ## EBE00229_S1

# %% 
# ### s85

# %%
adata_85  = ad.read_h5ad("../EBE00229_S1/adata/s85_processed.h5ad")
adata_85
adata_85.obs = adata_85.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_85.obs['simplified_ID'] = "no_chase_3"
adata_85

# %%
theta = np.radians(290) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_85.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_85.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_85.obsm['spatial'] + np.array([-42500, 8500])
adata_85.obsm['spatial'] = shifted_coordinates

# %% 
# ### s65

# %%
adata_65  = ad.read_h5ad("../EBE00229_S1/adata/s65_processed.h5ad")
adata_65
adata_65.obs = adata_65.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_65.obs['simplified_ID'] = "no_chase_2"
adata_65


# %%
theta = np.radians(50) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_65.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_65.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_65.obsm['spatial'] + np.array([-31000, -20000])
adata_65.obsm['spatial'] = shifted_coordinates

# %% 
# ### s63

# %%
adata_63  = ad.read_h5ad("../EBE00229_S1/adata/s63_processed.h5ad")
adata_63
adata_63.obs = adata_63.obs.rename(columns={'mean_intensity_reddot': 'mean_intensity_seg', 'max_intensity_reddot': 'max_intensity_seg', 'min_intensity_reddot':'min_intensity_seg'})
adata_63.obs['simplified_ID'] = "no_chase_1"
adata_63

# %%
theta = np.radians(325) 

# Create rotation matrix
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Get current coordinates
coords = adata_63.obsm['spatial']

# Apply rotation (this rotates around origin [0,0])
rotated_coords = coords @ rotation_matrix.T  # or np.dot(coords, rotation_matrix.T)

# Update the coordinates
adata_63.obsm['spatial'] = rotated_coords

shifted_coordinates = adata_63.obsm['spatial'] + np.array([-32000, -12000])
adata_63.obsm['spatial'] = shifted_coordinates



# %%
# # Concatenate

# %%
adata_comb = ad.concat([adata_74, adata_75, adata_76, adata_58, adata_59, adata_60, adata_right, adata_big, adata_small, adata_bottom, adata_85, adata_63, adata_65], join="outer", label="dataset")
adata_comb

# %%
adata_comb.obs_names_make_unique()

# %%
adata_comb.var['Blank'] = adata_comb.var.index.str.startswith('Blank').astype(int)
filter = adata_comb.var['Blank'] != 1
adata_comb = adata_comb[:, filter]
adata_comb

# %%
genes = pd.read_csv("../EBE_transcripts.csv", index_col=0)
genes

# %%
adata_comb.var = adata_comb.var.join(genes,how="left")
adata_comb

# %%
# add unique gene name
adata_comb.var['unique_gene_name'] = adata_comb.var.index.astype(str) + "_" + adata_comb.var['gene_name'].astype(str)

# %%
sq.pl.spatial_scatter(adata_comb, shape=None, color=["dataset"],size=1)

# %%
info = pd.read_csv("../pilot_info.csv", index_col=0)
info

# %%
adata_comb.obs = adata_comb.obs.join(info, on='simplified_ID', how='left')

# %%
adata_comb

# %%
samples = adata_comb.obs["simplified_ID"].unique()

# %%
samples

# %%
split_by_column = 'simplified_ID'

# %%
for sample in samples:
    # Create a boolean mask for the current category
    mask = adata_comb.obs[split_by_column] == sample

    # Subset the AnnData object
    # Note: Subsetting creates a view, which is memory-efficient
    subset_adata = adata_comb[mask, :].copy() # .copy() creates a new AnnData object, not a view

    # Define the output filename
    output_filename = f"{sample.replace(' ', '_')}.h5ad"

    # Save the subset AnnData object
    subset_adata.write(output_filename)

    print(f"Saved {output_filename}")
