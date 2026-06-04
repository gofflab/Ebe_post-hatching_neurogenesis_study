#!/usr/bin/env python3

#%% imports
import scanpy as sc 
#import squidpy as sq
import anndata as ad
import spaco
import numpy as np 
import pandas as pd
import plotnine as pn 
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display
from matplotlib_inline.backend_inline import set_matplotlib_formats
IPython.display.set_matplotlib_formats = set_matplotlib_formats
import json
#import geopandas as gpd
#from shapely.geometry import shape
import math
import os

########
#%% Utility functions
########
seed = 102998
from plotnine import theme, element_rect, element_text, element_line, element_blank
pd.set_option('mode.copy_on_write', True)  # This might help with the view issue


def theme_black():
    """
    A custom plotnine theme with black backgrounds and white text.
    
    Returns:
        theme: A plotnine theme object with black backgrounds and white text
    """
    return theme(
        # Overall plot background
        plot_background=element_rect(fill='black', color='black'),
        
        # Panel (plot area) background
        panel_background=element_rect(fill='black', color='black'),
        
        # Panel grid lines
        panel_grid_major=element_line(color='#333333', size=0.5),
        panel_grid_minor=element_line(color='#1a1a1a', size=0.25),
        
        # Panel border
        panel_border=element_rect(color='white', fill='none', size=1),
        
        # Axis lines
        axis_line=element_line(color='white', size=0.5),
        
        # Axis ticks
        axis_ticks=element_line(color='white', size=0.5),
        
        # Axis text
        axis_text=element_text(color='white', size=10),
        axis_text_x=element_text(color='white', size=10),
        axis_text_y=element_text(color='white', size=10),
        
        # Axis titles
        axis_title=element_text(color='white', size=12, weight='bold'),
        axis_title_x=element_text(color='white', size=12, weight='bold'),
        axis_title_y=element_text(color='white', size=12, weight='bold'),
        
        # Plot title and subtitle
        plot_title=element_text(color='white', size=16, weight='bold', ha='left'),
        plot_subtitle=element_text(color='white', size=12, ha='left'),
        
        # Legend
        legend_background=element_rect(fill='black', color='white'),
        legend_text=element_text(color='white', size=10),
        legend_title=element_text(color='white', size=11, weight='bold'),
        legend_key=element_rect(fill='black', color='black'),
        
        # Strip (facet labels) for faceted plots
        strip_background=element_rect(fill='#333333', color='white'),
        strip_text=element_text(color='white', size=10, weight='bold'),
        strip_text_x=element_text(color='white', size=10, weight='bold'),
        strip_text_y=element_text(color='white', size=10, weight='bold'),
        
        # Remove some elements for cleaner look (optional)
        # axis_ticks_length=0,  # Remove tick marks
        # panel_grid_minor=element_blank(),  # Remove minor grid lines
    )

# Alternative minimal version with no grid lines
def theme_black_minimal():
    """
    A minimal version of the black theme with no grid lines.
    
    Returns:
        theme: A minimal plotnine theme object with black backgrounds and white text
    """
    return theme(
        # Overall plot background
        plot_background=element_rect(fill='black', color='black'),
        
        # Panel (plot area) background
        panel_background=element_rect(fill='black', color='black'),
        
        # Remove grid lines
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        
        # Panel border
        panel_border=element_blank(),
        
        # Axis lines
        axis_line=element_line(color='white', size=0.5),
        
        # Axis ticks
        axis_ticks=element_line(color='white', size=0.5),
        
        # Axis text
        axis_text=element_text(color='white', size=10),
        
        # Axis titles
        axis_title=element_text(color='white', size=12, weight='bold'),
        
        # Plot title and subtitle
        plot_title=element_text(color='white', size=16, weight='bold', ha='left'),
        plot_subtitle=element_text(color='white', size=12, ha='left'),
        
        # Legend
        legend_background=element_blank(),
        legend_text=element_text(color='white', size=10),
        legend_title=element_text(color='white', size=11, weight='bold'),
        legend_key=element_rect(fill='black', color='black'),
        
        # Strip (facet labels)
        strip_background=element_rect(fill='#333333', color='white'),
        strip_text=element_text(color='white', size=10, weight='bold'),
    )

# Plot the cells in x,y position (from AnnData's adata.obs) and color them by their expression of a gene in [genes].  Facet by gene.
def plot_cells(adata,color='leiden',x='x',y='y',size=0.05,alpha=1,palette=None):
    # Convert adata to a pandas dataframe
    df = adata.obs
    df = df.reset_index()
    df = df.rename(columns={'index':'cell'})
    
    p = (
        pn.ggplot(df, pn.aes(x=x, y=y)) 
        + pn.geom_point(pn.aes(color=color),size=size,alpha=alpha)
        + theme_black_minimal()
        + pn.coord_equal()
    )
    if palette is not None:
        p += pn.scale_color_manual(values=palette)
    return p

#%%
sc.set_figure_params(facecolor="white", 
                     format='pdf', 
                     frameon=False, 
                     dpi=300, 
                     #figsize=[20,18],
                     figsize=[8,8],
                     vector_friendly=False,
                )
sc.settings.verbosity = 3
plt.rcParams["axes.grid"] = False

#%% Load datasets
datadir = '/Users/loyalgoff/Library/CloudStorage/GoogleDrive-loyalgoff@gmail.com/My Drive/Work/Goff Lab/Projects/Active/Eberryi/neurogenesis/20250925_v1_fixed_for_preprint/cleaned/'

# List all h5ad files in datadir
# h5ads = [x for x in os.listdir(datadir) if x.endswith('.h5ad')]

# #%%
# adatas = [sc.read_h5ad(datadir + h5ad) for h5ad in h5ads]

# # %%
# adata = ad.concat(adatas, label='section', index_unique=None)

#%% instead, use the pre-combined dataset
adata = sc.read_h5ad(datadir + 'combined/251006_combined_all_raw.h5ad')

#%%
# create new columns in adata.obs called 'x_adjusted' and 'y_adjusted' that use the values in adata.obsm['spatial']
adata.obs['x_adjusted'] = adata.obsm['spatial'][:,0]
adata.obs['y_adjusted'] = adata.obsm['spatial'][:,1]*-1

# %%
# plot cells by section but keep the axis coordinates equal
sc.pl.scatter(adata, x='x_adjusted', y='y_adjusted', color='section', size=1, frameon=False, show=False)

# %%
adata.layers['preprocessed'] = adata.X.copy()


# # %%
# # Translate coordinates to array sections
# # For each section in adata.obs, calculate the mean adata.obs['x'] and adata.obs['y']
# section_means = adata.obs.groupby('section', observed=True)[['x', 'y']].mean()

# # %%
# # Adjust the x and y coordinates for each cell by translating them to a new mean to create a 3 x 2 grid of sections. 
# section_means['x'] = section_means['x'] - section_means['x'].mean()
# section_means['y'] = section_means['y'] - section_means['y'].mean()

# # %%
# # Create a mapping from section to new coordinates
# section_mapping = {
#     section: (row['x'], row['y']) for section, row in section_means.iterrows()
# }

# # %% Create an `x_adjusted` and `y_adjusted` column in adata.obs by adjusting the original values using the new coordinates
# adata.obs['x_adjusted'] = adata.obs.apply(lambda row: row['x'] - section_mapping[row['section']][0], axis=1)
# adata.obs['y_adjusted'] = adata.obs.apply(lambda row: row['y'] - section_mapping[row['section']][1], axis=1)    

# #%%
# sc.pl.scatter(adata, x='x_adjusted', y='y_adjusted', color='section', size=3, frameon=False, show=False)

# #%% 
# nudge_amt = 15000
# adata.obs.loc[adata.obs['Region'] == 'Dorsal', 'y_adjusted'] += nudge_amt

# #%%
# ages = ['1 Days',
#         '5 Days',
#         '8 Days',
#         '15 Days',]

# #%% Stratify horizontally by age
# for i in range(len(ages)):
#     #print(ages[i+1:])
#     adata.obs.loc[adata.obs['Age'].isin(ages[i:]), 'x_adjusted'] += nudge_amt

# # %%# Plot the adjusted coordinates
# sc.pl.scatter(adata, x='x_adjusted', y='y_adjusted', color='section', size=4, frameon=False, show=False)

# #%% sample-specific rotation values for each section
# rotation_angles = {
#     '0': 110,   
#     '1': 115,   
#     '2': 130,   
#     '3': 110,  
#     '4': -30,    
#     '5': 60,  
#     '7': -80,   
#     '8': 90,  
#     '9': -5,   
#     '10': -10,  
#     '11': 20,  
#     '12': 80, 
#     '13': -5,   
#     '14': 45,    
# }

# adata.obs['x_rotated'] = adata.obs['x_adjusted'].copy()
# adata.obs['y_rotated'] = adata.obs['y_adjusted'].copy()
# #%% Function to rotate points around the mean center of a given section
# def rotate_section(adata, section):
#     angle = rotation_angles.get(section, 0)  # Default to 0 if section not found
#     theta = np.radians(angle)
#     cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    
#     # Get the mean center of the section
#     section_data = adata[adata.obs['section'] == section]
#     if section_data.n_obs == 0:
#         return  # No data for this section, skip rotation
    
#     x_mean = section_data.obs['x_adjusted'].mean()
#     y_mean = section_data.obs['y_adjusted'].mean()
    
#     # Rotate each point in the section
#     def rotate_point(row):
#         x_shifted = row['x_adjusted'] - x_mean
#         y_shifted = row['y_adjusted'] - y_mean
#         x_rotated = x_shifted * cos_theta - y_shifted * sin_theta + x_mean
#         y_rotated = x_shifted * sin_theta + y_shifted * cos_theta + y_mean
#         return pd.Series({'x_rotated': x_rotated, 'y_rotated': y_rotated})
    
#     rotated_coords = section_data.obs.apply(rotate_point, axis=1)
#     #print(rotated_coords.head())
#     adata.obs.loc[section_data.obs.index, ['x_rotated', 'y_rotated']] = rotated_coords

# #%% Apply rotation to each section
# for section in adata.obs['section'].unique():
#     rotate_section(adata, section)

# #%%
# sc.pl.scatter(adata, x='x_rotated', y='y_rotated', color='section', size=4, frameon=False, show=False)

#%% split index to retrieve gene_id
adata.var['gene_id'] = [".".join(x.split(".")[:-1]) for x in adata.var.index]

#%% remove genes where adata.var.index begins with 'Blank'
adata = adata[:,~adata.var.index.str.startswith('Blank')].copy()

#%%
adata.var['transcript_id'] = adata.var.index.copy()
adata.var = adata.var.set_index('gene_id')

###############
# Merge most up to date var info
###############
#%% utility function to update var names with most recent annotation on baserow
def fetch_annotation(): # May take a minute or two to run
    import requests
    API_URL = "https://baserow.gofflab.org/api/database/rows/table/{table_id}/"
    API_KEY="ExsMTTH0itiyOkSfkGn2j1BFtzHwO0LE" 
    DATABASE_ID = 208
    TABLE_ID = 1032

    # Headers with API Key for authentication
    headers = {
        'Authorization': f'Token {API_KEY}',
    }
    # Initial request to get the first page of rows
    url = API_URL.format(table_id=TABLE_ID)
    params = {'page': 1,
              'user_field_names': 'true',
              'size':200}
    
    all_rows = []
    while url:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            rows = data.get('results', [])
            all_rows.extend(rows)  # Add rows to the list
            
            # Check if there's a next page
            url = data.get('next', None)  # If there's a next page, it will be included in the response
            params = None  # Once we've got the URL for the next page, no need for params anymore
        else:
            print(f"Error: {response.status_code} - {response.text}")
            break
    
    return(pd.DataFrame(all_rows))

#%%
vars = fetch_annotation()

vars = vars.set_index('gene_id')

#%%
tmp = pd.merge(adata.var,vars,how='left',left_index=True,right_index=True)

#%%
adata.var = tmp

#%% Rename gene_name_x to gene_name, and drop gene_name_y
adata.var = adata.var.rename(columns={'gene_name_x':'gene_name'})
adata.var = adata.var.drop(columns=['gene_name_y'], errors='ignore')

# Update 'unique_gene_name' column as a concatenation of the index and either 'Ochierchiae_name' if it is not missing, otherwise 'gene_name'.
adata.var['unique_gene_name'] = adata.var['gene_name'] + '_' + adata.var.index

################
# Preprocessing
################

#%% Plot (plotnine) the distribution of n_genes_by_counts for all cells
p = (pn.ggplot(adata.obs, pn.aes(x='n_genes_by_counts'))
        + pn.geom_histogram(bins=100, fill='green')
)

p 

#%% Plot (plotnine) the distribution of total_counts for all cells
p = (pn.ggplot(adata.obs, pn.aes(x='total_counts'))
        + pn.geom_histogram(bins=100, fill='blue')
)

p

#%%
min_counts = 25
max_counts = 2500

#%% Filter cells and genes
sc.pp.filter_cells(adata, min_counts=min_counts)
sc.pp.filter_cells(adata, max_counts=max_counts)
sc.pp.filter_genes(adata, min_cells=5)

#%%
adata.layers["counts"] = adata.X.copy()

#%%
# Normalizing to median total counts
sc.pp.normalize_total(adata)

#%%
# Logarithmize the data
adata.layers['log1p'] = np.log1p(adata.X)

#%%
sc.pp.highly_variable_genes(adata,
                            min_disp=0.025,
                            layer='log1p')

sc.pl.highly_variable_genes(adata)

#%%
sc.tl.pca(adata,
          n_comps=80,
          svd_solver='arpack',
          random_state=seed,
          #use_highly_variable=True,
         )

#%%
sc.pl.pca_variance_ratio(adata, n_pcs=80, log=True)

#%%
sc.pp.neighbors(adata, 
                random_state=seed, 
                n_pcs=50, 
                #n_neighbors=25,
                metric='cosine')

#%%
sc.tl.umap(adata, 
           random_state=seed,
           min_dist=0.15, 
           )

#%%
sc.pl.umap(
    adata,
    color="roi",
    # Setting a smaller point size to prevent overlap
    size=4,
    frameon=False,
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=True,
    )

#%%
##############
# QC plots
##############
sc.pp.calculate_qc_metrics(
    adata, inplace=True, log1p=True, layer='preprocessed'
)

sc.pl.scatter(adata, "total_counts", "n_genes_by_counts",alpha=1,size=10)

#%%
# sc.pl.violin(
#     adata,
#     ["n_genes_by_counts", "total_counts"],
#     jitter=0.4,
#     multi_panel=True,
# )

# %%
for res in [0.5, 1.0, 2.0, 3.0]:
    sc.tl.leiden(
        adata, key_added=f"leiden_res_{res:4.2f}", resolution=res, random_state=seed, flavor="igraph",n_iterations=2
    )

#%%
sc.pl.umap(
    adata,
    color="leiden_res_2.00",
    # Setting a smaller point size to prevent overlap
    size=5,
    add_outline=True,
    legend_loc="on data",
    legend_fontsize="large",
)

#%%
adata.obsm['spatial'] = adata.obs[['x_adjusted', 'y_adjusted']].values

#%%
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="leiden_res_2.00",
    size=3,
    
)

##########
## Run up to here
##########
#%%
color_mapping_res2 = spaco.colorize(
    cell_coordinates=adata.obsm['spatial'],
    cell_labels=adata.obs['leiden_res_2.00'],
    colorblind_type="none",
    radius=0.3,
    n_neighbors=30,
    # palette=None, # when `palette` is not available, Spaco applies an automatic color selection
)

#%%
color_mapping_res2 = {k: color_mapping_res2[k] for k in adata.obs['leiden_res_2.00'].cat.categories}

#%%
palette_res2 = list(color_mapping_res2.values())

# #%%
# color_mapping_res1 = spaco.colorize(
#     cell_coordinates=adata.obsm['spatial'],
#     cell_labels=adata.obs['leiden_res_1.00'],
#     colorblind_type="none",
#     radius=0.3,
#     n_neighbors=30,
#     # palette=None, # when `palette` is not available, Spaco applies an automatic color selection
# )

# #%%
# color_mapping_res1 = {k: color_mapping_res1[k] for k in adata.obs['leiden_res_1.00'].cat.categories}

# #%%
# palette_res1 = list(color_mapping_res1.values())

# # %%
# sc.set_figure_params(facecolor="black",
#                      format='pdf', 
#                      frameon=False, 
#                      dpi=300, 
#                      figsize=[20,18],
#                      vector_friendly=False)

# #%%
# # Set axis label font color to white
# col = 'white'
# params = {"text.color" : col,
#           "ytick.color" : col,
#           "xtick.color" : col,
#           "axes.labelcolor" : col,
#           "axes.edgecolor" : "none",
#           "axes.facecolor" : "none",
#           "grid.color": "none",
#           }
# plt.rcParams.update(params)


# %%
sc.pl.umap(
    adata,
    color=["leiden_res_2.00"],
    legend_loc="on data",
    legend_fontsize="large",
    palette=palette_res2,
    size=2,
    add_outline=True,
    save="leiden_2.0_clusters.pdf",)

#%%
# p = plot_cells(adata, 
#            color='leiden_res_0.50', 
#            x='x_adjusted', 
#            y='y_adjusted',
#            palette=palette_res05,
#            size=0.005)

# p = (p + 
#     theme(figure_size=(20, 12),
#         legend_key_size=25,
#         legend_text=element_text(size=20),
#         legend_title=element_text(size=20)) + 
#     pn.guides(
#         color=pn.guide_legend(
#         override_aes={'size': 5},     # Point size in legend
#         )
#     )
# )
# p.draw()

# %%
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="leiden_res_2.00",
    size=2,
    save="leiden_2.0_clusters_spatial.pdf"
)

# %%
for i in adata.obs['leiden_res_2.00'].cat.categories:
    sc.pl.scatter(
        adata,
        x='x_adjusted',
        y='y_adjusted',
        color="leiden_res_2.00",
        size=2,
        groups=[i],
        palette=palette_res2,
        save=f"leiden_2.0_clusters_spatial_{i}.pdf",
    )

#%%
sc.tl.rank_genes_groups(adata, 'leiden_res_2.00', method='wilcoxon',layer="log1p") #max_iter=500 if using logreg

# %%
sc.set_figure_params(facecolor="white", format='pdf', frameon=False, dpi=300, figsize=[16,16])

# Set axis label font color to white
col = 'black'
params = {"text.color" : col,
          "ytick.color" : col,
          "xtick.color" : col,
          "axes.labelcolor" : "none",
          "axes.edgecolor" : "none",
}
plt.rcParams.update(params)
plt.rcParams["axes.grid"] = False

#%%
sc.pl.rank_genes_groups_dotplot(
    adata, groupby="leiden_res_2.00", standard_scale="var", n_genes=20,save='leiden_res_2.00.pdf', gene_symbols='unique_gene_name'
)

#%%
sc.set_figure_params(facecolor='black',format='pdf', frameon=False, dpi=300, figsize=[12,12])

col="white"
params = {"text.color" : col,
          "ytick.color" : col,
          "xtick.color" : col,
          "axes.labelcolor" : "white",
          "axes.facecolor" : "black",
          "axes.edgecolor" : "black",
}
plt.rcParams.update(params)
plt.rcParams["axes.grid"] = False

# %%
gene_id = ['EB45560','EB46007'] # Ascl1-1, Ascl1-2
gene_id = "EB07294" #Sevenup
gene_id = 'EB17597' #Runx1
gene_id = 'EB17694' #Nkx2-1
gene_id = 'EB10890' #Nkx2-5
# gene_id = 'EB08075' #Ngn1
# gene_id = 'EB29273' #Elav1
# gene_id = 'EB18941' #Erbb4
# #gene_id = 'EB44903' #Eaat
# #gene_id = 'EB25117' #Apolpp-8?
# #gene_id = 'EB32316' #Six4
# gene_id = 'EB22657' #Sox2
# gene_id = 'EB23886' #Pax6
# gene_id = 'EB08075' #Ngn1

sc.pl.umap(
    adata,
    color=gene_id,
    # Setting a smaller point size to prevent overlap
    size=8,
    add_outline=False,
    layer="log1p",
    #save = f'_{gene_id}_umap.pdf'
)

sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color=gene_id,
    size=4,
    layers="log1p",
    #save=f"_{gene_id}_spatial.pdf"
)
# %%
#Export for cellxgene
adata.obsm['X_spatial'] = adata.obs[['x_adjusted', 'y_adjusted']].values

# drop 'Cephexplorer_link' column from adata.var
adata.var = adata.var.drop(columns=['Cephexplorer_link'], errors='ignore')

bdata = adata.copy()
bdata.X = bdata.layers['log1p'].copy()

#%%
bdata.write_h5ad('20251007_loyal_analysis_with_EdU.h5ad')

#%%
########################
# Neighborhood analysis
########################
# import squidpy as sq

# #%%
# sq.gr.spatial_neighbors(adata, 
#                         coord_type="generic",
#                         spatial_key='spatial')

# #%%
# sq.gr.nhood_enrichment(adata, cluster_key="leiden_res_1.00", n_perms=1000)

# #%%
# sq.pl.nhood_enrichment(
#     adata, cluster_key="leiden_res_1.00", method="single", cmap="inferno", vmin=-50, vmax=100, save="neighborhood_leiden_1.00.pdf"
# )

# # %%
# sq.gr.spatial_autocorr(adata, mode="moran")
# adata.uns["moranI"].head(10)
# # %%
# sc.pl.scatter(
#     adata,
#     x='x_adjusted',
#     y='y_adjusted',
#     color="EB15884",
#     size=2,
#     layers="log1p",
# )

#################
#%% Playing with EdU intensities
#################
# adata.obs['log_edu'] = np.log1p(adata.obs['edu_mean_intensity'])

# #%%
# sc.pl.violin(adata,
#                 ['log_edu'],
#                 groupby='leiden_res_1.00',
#                 jitter=0.4,
                
#     )
# # %%
# sc.pl.scatter(
#     adata,
#     x='x_adjusted',
#     y='y_adjusted',
#     color="log_edu",
#     size=2,
#     frameon=False,
#     palette="magma",
#     #add_outline=True,
#     #save="log_edu_spatial.pdf"
# )

# # %% plot log_edu histogram by roi
# p = (
#     pn.ggplot(adata.obs, pn.aes(x='edu_mean_intensity', fill='roi')) 
#     + pn.geom_histogram(bins=50, alpha=0.7, position='identity')
#     + theme_black()
#     + pn.labs(title='EdU Intensity Distribution by roi', x='Log EdU Intensity', y='Count')
#     + pn.theme(figure_size=(10, 6))
#     + pn.facet_wrap('~roi', ncol=3, scales='free')  # Facet by roi
# )

# p.draw()

# %%
# Using the per-roi mode of 'edu_mean_intensity', select those cells for each roi that are 2 standard deviations above the mode
# def select_high_edu_cells(adata, threshold=2):
#     """
#     Select cells with EdU intensity above a certain threshold (default: 2 standard deviations above the mean).
    
#     Parameters:
#         adata (AnnData): The annotated data object containing 'edu_mean_intensity'.
#         threshold (float): The number of standard deviations above the mean to use as a cutoff.
        
#     Returns:
#         AnnData: A new AnnData object with only the selected cells.
#     """
#     # Calculate the mean and standard deviation of 'edu_mean_intensity' for each roi
#     edu_stats = adata.obs.groupby('roi')['edu_mean_intensity'].agg(['mean', 'std']).reset_index()
    
#     # Merge the stats back into adata.obs
#     adata.obs = adata.obs.merge(edu_stats, on='roi', suffixes=('', '_stats'))
    
#     # Create a mask for cells that are above the threshold
#     mask = adata.obs['edu_mean_intensity'] > (adata.obs['mean'] + threshold * adata.obs['std'])
    
#     # Return a new AnnData object with only the selected cells
#     return mask

# # %%
# mask = select_high_edu_cells(adata, threshold=2)

# # %%
# adata.obs['edu_positive'] = mask.to_list()

# # %%
# p = (
#     pn.ggplot(adata.obs, pn.aes(x='x_adjusted', y='y_adjusted', fill='edu_positive')) 
#     + pn.geom_point(size=0.75,stroke=0)
#     + theme_black()
#     + pn.scale_fill_manual(values=["#5F5F5F","#FF1414"], name='Edu Positive')
#     + pn.labs(title='edu_positive cells')
#     + pn.theme(figure_size=(28, 16))
#     + pn.coord_fixed()  # Keep the aspect ratio fixed
# )

# p.draw()

# #%%
# sc.set_figure_params(facecolor="black", format='pdf', frameon=False, dpi=300, figsize=[10, 10])

# # Set axis label font color to white
# col = 'w'
# params = {"text.color" : col,
#           "ytick.color" : col,
#           "xtick.color" : col,
#           "axes.labelcolor" : col,
#           "axes.edgecolor" : col}
# plt.rcParams.update(params)

# # %%
# sc.pl.umap(
#     adata,
#     color="edu_positive",
#     size=5,
#     frameon=False,
#     palette=["#5F5F5F","#FF1414"],
#     #add_outline=True,
#     #legend_loc="on data",
#     legend_fontsize="large",
#     save="_edu_positive_umap.pdf"
# )

# # %%
# # plot of the fraction of edu_positive cells for each leiden_res_1.0 cluster. Calculate the fraction of each cluster that is edu_positive first and then plot only that value

# edu_positive_fraction = adata.obs.groupby('leiden_res_1.00')['edu_positive'].mean().reset_index()

# p = (
#     pn.ggplot(edu_positive_fraction, pn.aes(x='leiden_res_1.00', y='edu_positive')) 
#     + pn.geom_bar(stat='identity', fill="#FF1414", alpha=0.7)
#     + theme_black()
#     + pn.labs(title='Fraction of edu_positive Cells by leiden_res_1.00 Cluster', x='leiden_res_1.00', y='Fraction of Cells')
#     + pn.theme(figure_size=(10, 6))
# )
# p.draw()

# %%
#############
# Celltype Annotation
#############

leiden_annotation = {
    '0': 'Unknown',
    '1': 'Sensory Epithelium?',
    '2': ,
    '3': ,
    '4': ,
    '5': ,
    '6': ,
    '7': ,
    '8': ,
    '9': ,
    '10': ,
    '11': ,
    '12': ,
    '13': ,
    '14': ,
    '15': 'White body neuronal progentiors',
    '16': ,
    '17': ,
    '18': ,
    '19': ,
    '20': ,
    '21': ,
    '22': ,
    '23': ,
    '24': ,
    '25': ,
    '26': ,
    '27': ,
    '28': ,
    '29': ,
    '30': ,
    '31': ,
    '32': ,
    '33': ,
    '34': ,
    '35': ,
    '36': 'White body neuronal progenitors',
}

#%% 
adata.obs['major_cell_type'] = adata.obs['leiden_res_1.00'].map(leiden_annotation)


# %%
sc.pl.umap(
    adata,
    color="major_cell_type",
    size=5,
    frameon=False,
    #add_outline=True,
    #legend_loc="on data",
    legend_fontsize="large",
    save="_major_cell_type_umap.pdf"
)

#########################
# Subsetting to neural identity celltypes
##########################
# %%
neural_clusters = ['1','2','5','7','9','10','11','14','15','16','17','19','20','21','22','23','24']
neural_adata = adata[adata.obs['leiden_res_1.00'].isin(neural_clusters)].copy()


#%%
sc.pp.highly_variable_genes(neural_adata,
                            min_disp=0.5,
                            layer='log1p')

sc.pl.highly_variable_genes(neural_adata)

# %%
sc.tl.pca(neural_adata,
          n_comps=50,
          svd_solver='arpack',
          random_state=seed,
          use_highly_variable=False,
         )

#%%
sc.set_figure_params(facecolor="white", 
                     format='pdf', 
                     frameon=False, 
                     dpi=300, 
                     #figsize=[20,18],
                     figsize=[8,8],
                     vector_friendly=False,
                )

sc.pl.pca_variance_ratio(neural_adata, n_pcs=50, log=True)

#%%
sc.pp.neighbors(neural_adata, 
                random_state=seed, 
                n_pcs=40, 
                n_neighbors=25,
                metric='cosine')

#%%
sc.tl.umap(neural_adata, 
           random_state=seed,
           min_dist=0.25, 
           )

#%%
sc.set_figure_params(facecolor="black", 
                     format='pdf', 
                     frameon=False, 
                     dpi=300, 
                     #figsize=[20,18],
                     figsize=[8,8],
                     vector_friendly=False,
                )

sc.pl.umap(
    neural_adata,
    color="major_cell_type",
    # Setting a smaller point size to prevent overlap
    size=4,
    frameon=False,
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=True,
    )

# %%
neural_adata.obs['edu_positive'] = neural_adata.obs['edu_positive'].astype('category')
neural_adata.write("20250827_neural_adata.h5ad")
# %%
