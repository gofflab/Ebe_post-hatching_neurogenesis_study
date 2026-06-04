# %% [markdown]
# ---
# title: "SOLARmap Analysis of E.berryi hatchlings"
# jupyter: python3
# ---
#%% imports
import scanpy as sc 
#import squidpy as sq
import anndata as ad
import spaco
import numpy as np 
import pandas as pd
import plotnine as pn 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import IPython.display
from matplotlib_inline.backend_inline import set_matplotlib_formats
IPython.display.set_matplotlib_formats = set_matplotlib_formats
import json
#import geopandas as gpd
#from shapely.geometry import shape
import math
import os
import string

########
#%% Utility functions
########
seed = 102998

sc.set_figure_params(vector_friendly=False)
sc.settings.verbosity = 3
figdir = "figures_for_resubmission"
sc.settings.figdir = figdir + "/"
plt.rcParams["axes.grid"] = False

from plotnine import theme, element_rect, element_text, element_line, element_blank
pd.set_option('mode.copy_on_write', True)  # This might help with the view issue

plt.rcParams['pdf.fonttype'] = 42

#####################
# Utility functions
#####################
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
def add_scale_bar(x_limits, y_limits,
                  um_per_pixel=0.216,  # Changed parameter name for clarity
                  color='black', 
                  size_um=100, 
                  position=('right', 'bottom'),
                  x_offset=0.05,
                  y_offset=0.05,
                  label=True,
                  linewidth=2):
    """
    Create a scale bar for spatial transcriptomics plots.
    
    Parameters
    ----------
    x_limits : tuple
        (min, max) values for x-axis
    y_limits : tuple
        (min, max) values for y-axis
    um_per_pixel : float, default=0.216
        Conversion factor: micrometers per pixel (how many µm in 1 pixel)
    color : str, default='black'
        Color of the scale bar
    size_um : float, default=100
        Size of the scale bar in micrometers
    position : tuple, default=('right', 'bottom')
        Position of scale bar. First element: 'left' or 'right'
        Second element: 'top' or 'bottom'
    x_offset : float, default=0.05
        Horizontal offset from edge as fraction of plot width
    y_offset : float, default=0.05
        Vertical offset from edge as fraction of plot height
    label : bool, default=True
        Whether to show the size label
    linewidth : float, default=2
        Width of the scale bar line
    
    Returns
    -------
    list
        List of plotnine annotation objects to add to a plot
    
    Example
    -------
    # If 1 pixel = 0.216 µm, then for a 100 µm scale bar:
    # bar_length = 100 / 0.216 = 463 pixels
    
    x_lim = (adata.obs['x_coord'].min(), adata.obs['x_coord'].max())
    y_lim = (adata.obs['y_coord'].min(), adata.obs['y_coord'].max())
    
    for layer in add_scale_bar(x_lim, y_lim, um_per_pixel=0.216, size_um=100):
        p = p + layer
    """
    
    # Calculate scale bar length in pixels
    # If 1 pixel = um_per_pixel µm, then size_um µm = size_um / um_per_pixel pixels
    bar_length_pixels = size_um / um_per_pixel
    
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate position
    if position[0] == 'right':
        x_start = x_max - x_offset * x_range - bar_length_pixels
    else:  # left
        x_start = x_min + x_offset * x_range
    
    if position[1] == 'bottom':
        y_pos = y_min + y_offset * y_range
    else:  # top
        y_pos = y_max - y_offset * y_range
    
    x_end = x_start + bar_length_pixels
    
    layers = [
        pn.annotate('segment', x=x_start, y=y_pos, xend=x_end, yend=y_pos,
                color=color, size=linewidth)
    ]
    
    if label:
        # Format label
        if size_um >= 1000:
            label_text = f'{size_um/1000:.0f} mm'
        else:
            label_text = f'{size_um:.0f} µm'
        
        # Position label above the bar
        label_y = y_pos + 0.02 * y_range
        label_x = (x_start + x_end) / 2
        
        layers.append(
            pn.annotate('text', x=label_x, y=label_y, 
                    label=label_text, 
                    color=color, 
                    size=10,
                    ha='center', va='bottom')
        )
    
    return layers

#%% scale bar for matplotlib axes
def add_scale_bar_mpl(ax, 
                      um_per_pixel=0.216,
                      color='black', 
                      size_um=100, 
                      position=('right', 'bottom'),
                      x_offset=0.05,
                      y_offset=0.05,
                      label=True,
                      linewidth=2,
                      fontsize=10):
    """
    Add a scale bar to a matplotlib axis (compatible with scanpy plots).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis object to add the scale bar to
    um_per_pixel : float, default=0.216
        Conversion factor: micrometers per pixel (how many µm in 1 pixel)
    color : str, default='black'
        Color of the scale bar
    size_um : float, default=100
        Size of the scale bar in micrometers
    position : tuple, default=('right', 'bottom')
        Position of scale bar. First element: 'left' or 'right'
        Second element: 'top' or 'bottom'
    x_offset : float, default=0.05
        Horizontal offset from edge as fraction of plot width
    y_offset : float, default=0.05
        Vertical offset from edge as fraction of plot height
    label : bool, default=True
        Whether to show the size label
    linewidth : float, default=2
        Width of the scale bar line
    fontsize : float, default=10
        Font size for the label
    
    Returns
    -------
    None
        Modifies the axis in place
    
    Example
    -------
    import scanpy as sc
    
    ax = sc.pl.spatial(adata, color='cell_type', show=False)
    add_scale_bar_mpl(ax, um_per_pixel=0.216, size_um=100)
    plt.show()
    
    # Or with embedding/scatter:
    ax = sc.pl.scatter(adata, x='x_coord', y='y_coord', color='cell_type', show=False)
    add_scale_bar_mpl(ax, um_per_pixel=0.216, size_um=50)
    plt.show()
    """
    
    # Calculate scale bar length in pixels
    bar_length_pixels = size_um / um_per_pixel
    
    # Get current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    # Calculate position
    if position[0] == 'right':
        x_start = x_max - x_offset * x_range - bar_length_pixels
    else:  # left
        x_start = x_min + x_offset * x_range
    
    if position[1] == 'bottom':
        y_pos = y_min + y_offset * y_range
    else:  # top
        y_pos = y_max - y_offset * y_range
    
    x_end = x_start + bar_length_pixels
    
    # Draw the scale bar
    ax.plot([x_start, x_end], [y_pos, y_pos], 
            color=color, linewidth=linewidth, 
            solid_capstyle='butt', zorder=1000)
    
    # Add label if requested
    if label:
        # Format label
        if size_um >= 1000:
            label_text = f'{size_um/1000:.0f} mm'
        else:
            label_text = f'{size_um:.0f} µm'
        
        # Position label above the bar
        label_y = y_pos + 0.015 * y_range
        label_x = (x_start + x_end) / 2
        
        ax.text(label_x, label_y, label_text,
                color=color, fontsize=fontsize,
                ha='center', va='bottom', zorder=1000)
    
    return ax

#%%
def plot_gene(adata,gene_id,x='x_adjusted',y='y_adjusted',size=0.05,alpha=1,use_gene_name=False,layer=None):
    df = adata.obs
    df = df.reset_index()
    df = df.rename(columns={'index':'cell'})
    # if gene_id is in adata.obs, use that instead of adata[:,gene_id].X
    if gene_id in adata.obs.columns:
        pass
    else:
        if layer is not None:
            df[gene_id] = adata[:,gene_id].layers[layer].toarray().flatten()
        else:
            df[gene_id] = adata[:,gene_id].X.toarray().flatten()
        if use_gene_name:
            gene_name = adata.var.loc[gene_id,'unique_gene_name']
    p = (
        pn.ggplot(df, pn.aes(x=x, y=y)) 
        + pn.geom_point(pn.aes(color=gene_id),size=size,alpha=alpha)
        + theme_black_minimal()
        + pn.coord_equal()
        + pn.scale_color_cmap('viridis', na_value="#333333")
    )
    if use_gene_name and gene_id not in adata.obs.columns:
        p += pn.ggtitle(f"{gene_name}")
    else:
        p += pn.ggtitle(f"{gene_id}")
    return p

#%%
def white_fig():
    sc.set_figure_params(facecolor='white',
                         format='pdf', 
                         frameon=False, 
                         dpi=300, 
                         figsize=[8,8],
                         vector_friendly=False)

    col="black"
    params = {"text.color" : col,
            "ytick.color" : col,
            "xtick.color" : col,
            "axes.labelcolor" : "black",
            "axes.facecolor" : "white",
            "axes.edgecolor" : "black",
    }
    plt.rcParams.update(params)
    plt.rcParams["axes.grid"] = False
    return
#%%
def black_fig(figsize=[12,12]):
    sc.set_figure_params(facecolor='black',
                         format='pdf', 
                         frameon=False, 
                         dpi=300, 
                         figsize=figsize,
                         vector_friendly=False)
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
    return

black_fig()

#%% Load datasets
datadir = 'cleaned/'

# List all h5ad files in datadir
h5ads = [x for x in os.listdir(datadir) if x.endswith('.h5ad')]
h5ads.sort()

samples = [x.rsplit('.', 1)[0] for x in h5ads]

# Get annotation files
annotation_files = [datadir + "cellxgene_annotation/" + x + "_loyal_annotation.csv" for x in samples]


#%%
adatas = [sc.read_h5ad(datadir + h5ad) for h5ad in h5ads]

#%% Parse annotations and add to adata.obs
for i in range(len(samples)):
    print(f"Processing {samples[i]}...")
    annots = pd.read_csv(annotation_files[i], index_col=0, comment='#')
    tmp = adatas[i].obs.join(annots, how='left')
    adatas[i].obs = tmp
    print(f"\tFinished processing {samples[i]}.")

# %%
adata = ad.concat(adatas, label='section', index_unique=None)

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

# Drop 'Cephexplorer_link' column in adata.var
adata.var = adata.var.drop(columns=['Cephexplorer_link'], errors='ignore')

#%% Create column 'section_name' in adata.obs mapping adata.obs['section'] to samples
section_map = {f"{i}": samples[i] for i in range(len(samples))}
adata.obs['section_name'] = adata.obs['section'].map(section_map)

#%% Change 'Lens' to 'Cornea' in adata.obs['Tissue'] to correct prior misannotation
adata.obs['Tissue'] = adata.obs['Tissue'].replace({'Lens': 'Cornea'})

#%%
##############
# Fix orientation for upside down sections
###############
def flip_y_coordinates(adata, sections):
    for section in sections:
        mask = adata.obs['section_name'] == section
        y_coords = adata.obs.loc[mask, 'y_adjusted']
        max_y = y_coords.max()
        min_y = y_coords.min()
        adata.obs.loc[mask, 'y_adjusted'] = max_y + min_y - y_coords
    return adata

#%% Apply the function to flip y coordinates for specified sections
sections_to_flip = ['14day_2','14day_3','14day_4']

adata = flip_y_coordinates(adata, sections_to_flip)

#%% Shift 14day_2 down by 1250 in y and 14day_3 down by 1000 in y
adata.obs.loc[adata.obs['section_name'] == '14day_2', 'y_adjusted'] -= 1250
adata.obs.loc[adata.obs['section_name'] == '14day_3', 'y_adjusted'] -= 1000

#%% switch values of 'white_body_anterior' and 'white_body_posterior' for all sections in sections_to_flip
adata.obs.loc[adata.obs['section_name'].isin(sections_to_flip), 'White Body'] = adata.obs.loc[adata.obs['section_name'].isin(sections_to_flip), 'White Body'].replace({'white_body_anterior': 'white_body_posterior', 'white_body_posterior': 'white_body_anterior'})

# %%
adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_resubmission.h5ad')

#%%
################################
# Reprocess with pearson residuals
################################
#%%
adata.X = adata.layers['counts'].copy()

#%% calculate mean and total counts per gene
sc.pp.calculate_qc_metrics(adata, inplace=True, layer='counts')

#%%
adata.layers['sqrt_norm'] = np.sqrt(sc.pp.normalize_total(adata, inplace=False)["X"])

# %%
sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals",n_top_genes=400)

#%% Plot pearson residuals vs mean
p = (
    pn.ggplot(adata.var,pn.aes(x='mean_counts',y='residual_variances',color='highly_variable'))
    + pn.geom_point()
    + pn.theme_538()
    + pn.ggtitle("Pearson residuals")
    + pn.scale_color_manual(values=["#999999","#882200"])
    + pn.scale_x_log10()
    + pn.scale_y_log10()
)
p.show()

#%%
p.save(f"{figdir}/pearson_residuals_plot.pdf")

#%% find high variance pearson residual genes
sc.experimental.pp.normalize_pearson_residuals(adata)

# %%
sc.tl.pca(adata,
          svd_solver='arpack',
          random_state=seed,
          n_comps=50,
          use_highly_variable=False
          )

white_fig()
sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50)

# %%
sc.pp.neighbors(adata, 
                random_state=seed, 
                n_pcs=30, 
                #n_neighbors=25,
                #metric='cosine'
                )

#%%
sc.tl.umap(adata,
           random_state=seed,
           min_dist=0.10,
           method='umap',
           spread=2,
           )


#%%
black_fig()
sc.pl.umap(adata,
            color='Tissue',
            frameon=False,
            # Setting a smaller point size to prevent overlap
            size=3,
            add_outline=True,
            legend_loc="on data",
            legend_fontsize="large",
            title='Tissue Clusters',
            save="_Tissue_pearson_residuals.pdf"
          )

#%% Recalculate leiden clusters at a variety of resolutions.
for res in [0.5, 1.0, 2.0, 3.0]:
    sc.tl.leiden(
        adata, key_added=f"leiden_res_{res:3.1f}", resolution=res, random_state=seed, flavor="igraph",n_iterations=2
    )

#%% optimize colors for leiden_res_1.0
best_res = 'leiden_res_1.0'

color_mapping = spaco.colorize(
    cell_coordinates=adata.obsm['spatial'],
    cell_labels=adata.obs[best_res],
    colorblind_type="none",
    radius=0.3,
    n_neighbors=30,
    # palette=None, # when `palette` is not available, Spaco applies an automatic color selection
)

#%%
color_mapping = {k: color_mapping[k] for k in adata.obs[best_res].cat.categories}

#%%
palette = list(color_mapping.values())

#%%
tissue_palette = ['#2d6f2a','#ffbbbb', '#377eb8', '#984ea3', '#ff7f00','#49aed8', '#e41a1c','#333333']

#%%
#################
# Orientation plots
#################


# black_fig()
# sc.pl.umap(
#     adata,
#     color="leiden_res_0.5",
#     # Setting a smaller point size to prevent overlap
#     size=3,
#     add_outline=True,
#     legend_loc="on data",
#     legend_fontsize="large",
#     save="_leiden_res_0.5_umap.pdf"
# )

#%%
black_fig()
sc.pl.umap(
    adata,
    color="leiden_res_1.0",
    # Setting a smaller point size to prevent overlap
    size=3,
    add_outline=True,
    legend_loc="on data",
    legend_fontsize="large",
    palette=palette,
    save="_leiden_res_1.0_umap.pdf"
)

# Noting here that under these conditions and parameter assumptions, leiden_res_1.0 clusters 24 and 29 are the 'fontes neurales' cells.

#%%
###############
# Separate White Body (hematopoietic) and fontes neurales cells
###############
mask = adata.obs['leiden_res_1.0'].isin(['24','29']).to_numpy()
adata.obs['fontes_neurales'] = mask
# convert to categorical
adata.obs['fontes_neurales'] = adata.obs['fontes_neurales'].astype('bool')

tissue = adata.obs['Tissue'].copy()
if 'Fontes Neurales' not in tissue.cat.categories:
    tissue = tissue.cat.add_categories(['Fontes Neurales'])
tissue[mask] = 'Fontes Neurales'
adata.obs['Tissue'] = tissue

wb = adata.obs['White Body'].copy()
wb[mask] = 'unassigned'
adata.obs['White Body'] = wb

#%% Save anndata with updated annotations
adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_resubmission.h5ad')


#%% Update tissue_palette to include 'Fontes Neurales'
tissue_palette = ['#2d6f2a','#ffbbbb', '#377eb8', '#984ea3', '#ff7f00','#49aed8', '#e41a1c','#333333', "#DCDC15"]  # Added a new color for 'Fontes Neurales'

#%%
black_fig()
sc.pl.umap(
    adata,
    color="Tissue",
    # Setting a smaller point size to prevent overlap
    size=3,
    add_outline=True,
    legend_loc="on data",
    legend_fontsize="large",
    palette=tissue_palette,
    save="_Tissue_umap.pdf"
)

#%% 
sc.pl.umap(adata,
            color='Tissue',
            frameon=False,
            # Setting a smaller point size to prevent overlap
            size=3,
            add_outline=True,
            legend_loc="on data",
            legend_fontsize="large",
            palette=tissue_palette,
            title='Tissue Clusters',
            save="_Tissue_pearson_residuals_white.pdf"
          )

#%%
black_fig()
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="leiden_res_1.0",
    size=2,
    save="_leiden_res_1.0_spatial.pdf"
    
)

#%% 
white_fig()
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Tissue",
    size=2,
    palette=tissue_palette,
    frameon=False,
    save="_Tissue_spatial_white.pdf"
    
)

#%% 
black_fig()
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Tissue",
    size=2,
    palette=tissue_palette,
    frameon=False,
    save="_Tissue_spatial_black.pdf"
    
)

#%%
black_fig()
optic_lobe_palette = ['#e41a1c', '#49aed8', '#2d6f2a', '#333333']
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Optic Lobe",
    size=2,
    palette=optic_lobe_palette,
    save="_Optic_Lobe_spatial.pdf",
    
)

#%%
brain_structure_palette = ['#e41a1c', '#377eb8', '#2d6f2a', '#984ea3', '#ff7f00', '#333333']
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Brain Structure",
    size=2,
    palette=brain_structure_palette,
    save="_Brain_Structure_spatial.pdf"
)

#%%
position_palette = ['#2d6f2a','#ffbbbb', '#377eb8',]
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="position",
    size=2,
    palette=position_palette,
    save="_position_spatial.pdf"
)

#%%
white_body_palette = ['#333333',"#9e3387","#e21d1d", '#377eb8']
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="White Body",
    size=2,
    palette=white_body_palette,
    save="_White_Body_spatial.pdf"
)

#%% Age
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="death_age_days",
    size=2,
    save="_Age_spatial.pdf"
)

#%% Make a spatial scatter for each leiden cluster
black_fig()

# Currently commented out because this just takes a long time to draw...
for leiden_cluster in adata.obs['leiden_res_1.0'].cat.categories:
    sc.pl.scatter(
        adata,
        x='x_adjusted',
        y='y_adjusted',
        color='leiden_res_1.0',
        groups=leiden_cluster,
        size=2,
        title=f"Leiden Cluster - {leiden_cluster}",
        save=f"_leiden_res_1.0_cluster_{leiden_cluster}_spatial.pdf"
    )
    
    sc.pl.umap(
        adata,
        color='leiden_res_1.0',
        groups=leiden_cluster,
        size=3,
        title=f"Leiden Cluster - {leiden_cluster}",
        save=f"_leiden_res_1.0_cluster_{leiden_cluster}_umap.pdf"
    )

# %% plotnine heatmap of the proportion of each leiden cluster in each Brain Structure
# cat_of_interest = 'Tissue'
# cat_of_interest = 'Brain Structure'
# cat_of_interest = 'Optic Lobe'
# cat_of_interest = 'White Body'
# cat_of_interest = 'fontes_neurales'

# df = (
#     adata.obs
#     .groupby(['leiden_res_1.0', cat_of_interest])
#     .size()
#     .reset_index(name='counts')
# )
# df_total = (
#     adata.obs
#     .groupby(['leiden_res_1.0'])
#     .size()
#     .reset_index(name='total_counts')
# )
# df = df.merge(df_total, on='leiden_res_1.0')
# df['proportion'] = df['counts'] / df['total_counts']

# # order leiden clusters by hierarchical clustering of proportions
# proportion_matrix = df.pivot(index='leiden_res_1.0', columns=cat_of_interest, values='proportion').fillna(0)
# from scipy.cluster.hierarchy import linkage, leaves_list
# linkage_matrix = linkage(proportion_matrix, method='ward')
# ordered_leiden = leaves_list(linkage_matrix)
# leiden_order = [proportion_matrix.index[i] for i in ordered_leiden]
# df['leiden_res_1.0'] = pd.Categorical(df['leiden_res_1.0'], categories=leiden_order)

# #
# p = (
#     pn.ggplot(df, pn.aes(x=cat_of_interest, y='leiden_res_1.0', fill='proportion'))
#     + pn.geom_tile()
#     + pn.scale_fill_cmap('viridis')
#     + theme_black()
#     + pn.theme(
#         axis_text_x=element_text(rotation=45, hjust=1),
#         figure_size=(6,4)
#     )
#     + pn.ggtitle(f"Proportion of Each Leiden Cluster in {cat_of_interest}")
# )

# p.show()

# #
# p.save(f"{figdir}/heatmap_leiden_vs_{cat_of_interest}.pdf", width=5, height=6)

#%%
with plt.rc_context():
    sc.pl.scatter(
        adata,
        x='x_adjusted',
        y='y_adjusted',
        color="section_name",
        size=2,
    )

    # Save figure as vector image
    plt.savefig(f"{figdir}/20260513_all_sections_spatial.pdf", format='pdf', bbox_inches="tight")

#%%
# Do the same plot but with plotnine
p = plot_cells(adata, 
               color='section_name', 
               x='x_adjusted', y='y_adjusted', 
               size=0.05, 
               alpha=1)
p += pn.scale_color_manual(values=sns.color_palette("tab20", n_colors=len(samples)).as_hex())
p += pn.ggtitle("All Sections")
p.save(f"{figdir}/20260513_all_sections_spatial_plotnine.pdf", width=25, height=25)

#%%
######################
# Neural Stem markers across annotations
#####################

#%%
neural_stem_genes = ['EB45560', #Ascl1-1
                     'EB46007', #Ascl1-2
                    #  'EB07294', #Sevenup
                     'EB17597', #Runx1
                     'EB17694', #Nkx2-1
                    #  'EB10890', #Nkx2-5
                     'EB08075', #Ngn1
                    #  'EB29273', #Elav1
                    # 'EB33473', #Elav4
                    #  'EB18941', #Erbb4
                    #  'EB44903', #Eaat
                    #  'EB25117', #Apolpp-8?
                     'EB32316', #Six4
                     'EB22657', #Sox2
                     'EB23886', #Pax6
                     #'EB06384',  #ID4
                     'EB32583', #Erm (Fezf)
                     'EB01471', #Ems (empty spiracles)
                     'EB20937', #vnd (nkx2-2)
                     #'EB31550', #vvl 
                     'EB19103', #Diachete
                     'EB28226', #Gsx1
                     #'EB46862', #pnt (pointed)
                     'EB56110', #Kruppel
                     'EB13627', # Kruppel-like
                     #'EB14920', #Grh (Grainyhead)
                     
                     ]

neural_stem_gene_symbols = adata.var.loc[neural_stem_genes, 'unique_gene_name'].tolist()

#%%
# Summary expression of gene across 'Tissue' annotation
sc.pl.matrixplot(
    adata,
    var_names=neural_stem_gene_symbols,
    groupby='Tissue',
    layer="log1p",
    cmap='viridis',
    standard_scale='var',
    gene_symbols='unique_gene_name',
    dendrogram=True,
    save=f"_matrixplot_Tissue.pdf"
)

# Summary expression of gene across 'leiden_res_1.00' annotation
sc.pl.matrixplot(
    adata,
    var_names=neural_stem_gene_symbols,
    groupby='leiden_res_1.0',
    layer="log1p",
    cmap='viridis',
    standard_scale='var',
    gene_symbols='unique_gene_name',
    dendrogram=True,
    save=f"_matrixplot_leiden_res_1.00.pdf"
)

# %%
# gene_id = ['EB45560'] # Ascl1-1
# gene_id = ['EB46007'] # Ascl1-2
# gene_id = "EB07294" #Sevenup
# gene_id = 'EB17597' #Runx1
# gene_id = 'EB17694' #Nkx2-1
# gene_id = 'EB10890' #Nkx2-5
# gene_id = 'EB08075' #Ngn1
# gene_id = 'EB29273' #Elav1
# gene_id = 'EB18941' #Erbb4
# gene_id = 'EB44903' #Eaat
# gene_id = 'EB25117' #Apolpp-8?
# gene_id = 'EB32316' #Six4
# gene_id = 'EB22657' #Sox2
# gene_id = 'EB23886' #Pax6
# gene_id = 'EB08075' #Ngn1
# gene_id = 'EB06384' #ID4
# gene_id = 'EB14920' #Grh (Grainyhead)
# gene_id = 'EB02432' #Foxg1
# gene_id = 'EB39234' # PCNA
# gene_id = 'EB50310' # Snai2 (snail)
# gene_id = 'EB44903' # Eaat
# gene_id = 'EB31550' # vvl
#gene_id = 'EB29593' # Lhx3
gene_id = 'EB29595' # Elav3

black_fig()

sc.pl.umap(
    adata,
    color=gene_id,
    # Setting a smaller point size to prevent overlap
    size=8,
    add_outline=False,
    layer="log1p",
    save = f'_{gene_id}_umap.pdf'
)

sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color=gene_id,
    size=4,
    layers="log1p",
    save=f"_{gene_id}_spatial.pdf"
)


# Violin plot of gene expression across 'Tissue' annotation
# sc.pl.violin(
#     adata,
#     keys=gene_id,
#     groupby='Tissue',
#     layer="log1p",
#     size=2,
#     color='Tissue',
#     #cut=3,
#     log=False,
#     save=f"_{gene_id}_violin_Tissue.pdf"
# )


#%%
##################
# Scanpy differential tests
##################
obs_to_test = ['Tissue', 'leiden_res_0.5', 'leiden_res_1.0']

for obs in obs_to_test:
    sc.tl.rank_genes_groups(
        adata,
        groupby=obs,
        method='wilcoxon',
        layer='log1p',
        key_added=f'rank_{obs}'
    )

#%%
white_fig()

for obs in obs_to_test:
    sc.pl.rank_genes_groups(
        adata,
        key=f'rank_{obs}',
        n_genes=25,
        sharey=False,
        gene_symbols='unique_gene_name',
        save=f'_{obs}_top25.pdf',
        
    )
    
    sc.pl.rank_genes_groups_heatmap(
        adata,
        key=f'rank_{obs}',
        n_genes=10,
        swap_axes=True,
        layer='log1p',
        standard_scale='var',
        cmap='viridis',
        show_gene_labels=True,
        gene_symbols='unique_gene_name',
        dendrogram=True,
        save=f'_{obs}_heatmap_top10.pdf'
    )

#%% Heatmap of top n differential genes from leiden_res_0.5
n_genes = 5
sc.pl.rank_genes_groups_heatmap(
    adata,
    key='rank_leiden_res_0.5',
    n_genes=n_genes,
    swap_axes=True,
    layer='log1p',
    standard_scale='var',
    cmap='viridis',
    show_gene_labels=True,
    gene_symbols='unique_gene_name',
    dendrogram=True,
    save=f'_leiden_res_0.5_heatmap_top10.pdf'
)

#%%
black_fig()
Ascl1_scatter = plot_gene(adata,
              gene_id='EB45560', #Ascl1-1
              x='x_adjusted',
              y='y_adjusted',
              size=0.05,
              alpha=1,
              layer='log1p',
              )

x_lim = (adata.obs['x_adjusted'].min(), adata.obs['x_adjusted'].max())
y_lim = (adata.obs['y_adjusted'].min(), adata.obs['y_adjusted'].max())

for layer in add_scale_bar(x_lim, y_lim, color='white', size_um=1000):
    Ascl1_scatter = Ascl1_scatter + layer

Ascl1_scatter.save(f"{figdir}/gene_callouts/Ascl1_1_spatial_with_scale_bar.pdf", width=25, height=25)
#%%
#####################
# White Body Only
#####################
wb_adata = adata[adata.obs['Tissue'] == 'White Body'].copy()

#%%
######################
# Fontes neurales only
######################
fn_adata = adata[adata.obs['fontes_neurales']].copy()
# Set fontes_neurales to categorical
fn_adata.obs['fontes_neurales'] = fn_adata.obs['fontes_neurales'].astype('category')

#%%
black_fig()

sc.pl.scatter(
    fn_adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Tissue",
    size=3,
    save="_Fontes_Neurales_only_spatial.pdf"
)

# %%
# Get a table of counts of each leiden cluster in wb_adata.obs['leiden_res_1.0']
# This should be cleanly 24 and 29 because that's how subset was defined.
leiden_counts = fn_adata.obs['leiden_res_1.0'].value_counts()
print(leiden_counts)

#%%
################
#  Preprocessing fn subset
################
min_counts = 10
max_counts = 2500

#%% Filter cells and genes
sc.pp.filter_cells(fn_adata, min_counts=min_counts)
sc.pp.filter_cells(fn_adata, max_counts=max_counts)
sc.pp.filter_genes(fn_adata, min_cells=5)

#%%
#sc.pp.highly_variable_genes(fn_adata,
#                            min_disp=0.01,
#                            layer='log1p')

# white_fig()
# sc.pl.highly_variable_genes(wb_adata)

#%%

# sc.tl.pca(wb_adata,
#           n_comps=40,
#           svd_solver='arpack',
#           random_state=seed,
#           use_highly_variable=True,
#          )

# #%%
# sc.pl.pca_variance_ratio(wb_adata, n_pcs=40, log=True)

# #%%
# sc.pp.neighbors(wb_adata, 
#                 random_state=seed, 
#                 n_pcs=15, 
#                 #n_neighbors=25,
#                 metric='cosine')

# #%%
# sc.tl.umap(wb_adata, 
#            random_state=seed,
#            min_dist=0.15,
#            method='umap'
#            )

#%%
sc.tl.pca(fn_adata,
          svd_solver='arpack',
          random_state=seed,
          n_comps=50,
          use_highly_variable=False
          )

white_fig()
sc.pl.pca_variance_ratio(fn_adata, log=True, n_pcs=50)

# %%
sc.pp.neighbors(fn_adata, 
                random_state=seed, 
                n_pcs=40, 
                n_neighbors=25,
                #metric='cosine'
                )

#%%
sc.tl.umap(fn_adata,
           random_state=seed,
           min_dist=0.10,
           method='umap',
           spread=2,
           )


#%%
black_fig(figsize=[8,10])
sc.pl.umap(
    fn_adata,
    color="Tissue",
    # Setting a smaller point size to prevent overlap
    size=10,
    frameon=False,
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=True,
    save="_Fontes_Neurales_only_umap.pdf"
    )

#%% Ascl1
sc.pl.umap(
    fn_adata,
    color="EB45560", #Ascl1
    # Setting a smaller point size to prevent overlap
    size=10,
    frameon=False,
    cmap='plasma',
    layer='log1p',
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=False,
    title="Ascl1 expression in Fontes Neurales cells",
    save="_Ascl1_umap_fontes_neurales_only.pdf"
    )

#%%
sc.pl.umap(
    fn_adata,
    color="EB10890", #Nkx2-5
    # Setting a smaller point size to prevent overlap
    size=10,
    frameon=False,
    cmap='plasma',
    layer='log1p',
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=False,
    title="Nkx2.5 expression in Fontes Neurales cells",
    save="_Nkx2-5_umap_fontes_neurales_only.pdf"
    )

#%%
sc.pl.umap(
    fn_adata,
    color="EB17694", #Nkx2-1
    size=10,
    frameon=False,
    cmap='plasma',
    layer='log1p',
    #legend_loc="on data",
    legend_fontsize="medium",
    add_outline=False,
    title="Nkx2.1 expression in Fontes Neurales cells",
    save="_Nkx2-1_umap_fontes_neurales_only.pdf"
    )

#%%
pNkx25 = plot_gene(fn_adata,
              gene_id='EB10890', #Nkx2-5
              x='x_adjusted',
              y='y_adjusted',
              size=0.05,
              alpha=1,
              )

x_lim = (fn_adata.obs['x_adjusted'].min(), fn_adata.obs['x_adjusted'].max())
y_lim = (fn_adata.obs['y_adjusted'].min(), fn_adata.obs['y_adjusted'].max())

for layer in add_scale_bar(x_lim, y_lim, color='white', size_um=1000):
    pNkx25 = pNkx25 + layer

pNkx25.show()

#%%
pNkx21 = plot_gene(fn_adata,
              gene_id='EB17694', #Nkx2-1
              x='x_adjusted',
              y='y_adjusted',
              size=0.05,
              alpha=1,
              )

for layer in add_scale_bar(x_lim, y_lim, color='white', size_um=1000):
    pNkx21 = pNkx21 + layer

pNkx21.show()
#%%
#(pNkx25 | pNkx21)

#%% Clustering
# %%
for res in [0.5, 1.0, 2.0]:
    sc.tl.leiden(
        fn_adata, key_added=f"fn_leiden_res_{res:3.1f}", resolution=res, random_state=seed, flavor="igraph",n_iterations=2
    )

#%%
sc.pl.umap(
    fn_adata,
    color="fn_leiden_res_0.5",
    # Setting a smaller point size to prevent overlap
    size=10,
    frameon=False,
    #legend_loc="on data",
    legend_fontsize="medium",
    title = "Fontes Neurales Only\nfn_leiden_res_0.5",
    add_outline=True,
    save="_fn_leiden_res_0.5_umap_fontes_neurales_only.pdf"
    )

#%%
sc.pl.umap(
    fn_adata,
    color="fn_leiden_res_1.0",
    # Setting a smaller point size to prevent overlap
    size=10,
    frameon=False,
    #legend_loc="on data",
    legend_fontsize="medium",
    title='Fontes Neurales Only\n fn_leiden_res_1.0',
    add_outline=True,
    save="_fn_leiden_res_1.0_umap_fontes_neurales_only.pdf"
    )

#%%
sc.pl.scatter(
    fn_adata,
    x='x_adjusted',
    y='y_adjusted',
    color="fn_leiden_res_0.5",
    size=3,
    save="_fn_leiden_res_0.5_spatial_fontes_neurales_only.pdf"
)

################
# DE tests within fontes neurales only
################
# %%
obs_to_test = ['position', 'fn_leiden_res_0.5', 'fn_leiden_res_1.0']

#%%
for obs in obs_to_test:
    sc.tl.rank_genes_groups(
        fn_adata,
        groupby=obs,
        method='wilcoxon',
        layer='log1p',
        key_added=f'rank_{obs}'
    )

############################### Loyal edited to here 2026-05-13 ##################

#%%
white_fig()

for obs in obs_to_test:
    sc.pl.rank_genes_groups(
        fn_adata,
        key=f'rank_{obs}',
        n_genes=25,
        sharey=False,
        gene_symbols='unique_gene_name',
        save=f'_{obs}_top25_fontes_neurales_only.pdf',
    )
    
    sc.pl.rank_genes_groups_heatmap(
        fn_adata,
        key=f'rank_{obs}',
        n_genes=10,
        swap_axes=True,
        layer='log1p',
        standard_scale='var',
        cmap='viridis',
        show_gene_labels=True,
        gene_symbols='unique_gene_name',
        dendrogram=True,
        save=f'_{obs}_heatmap_top10_fontes_neurales_only.pdf',
    )

    sc.pl.rank_genes_groups_matrixplot(
        fn_adata,
        key=f'rank_{obs}',
        groupby=obs,
        n_genes=15,
        layer='log1p',
        standard_scale='var',
        cmap='viridis',
        gene_symbols='unique_gene_name',
        save=f'_{obs}_matrixplot_top15_fontes_neurales_only.pdf',
    )

#%% Heatmap of top n differential genes from leiden_res_0.5
n_genes = 5
sc.pl.rank_genes_groups_heatmap(
    fn_adata,
    key='rank_fn_leiden_res_0.5',
    n_genes=n_genes,
    swap_axes=True,
    layer='log1p',
    standard_scale='var',
    cmap='viridis',
    show_gene_labels=True,
    gene_symbols='unique_gene_name',
    dendrogram=True,
    save=f'_fn_leiden_res_0.5_heatmap_fn_only.pdf'
)

#%% WB gene plots
#gene_id = 'EB40727' # Lhx1
gene_id = 'EB32316' # Six4
gene_id = 'EB37598' # Foxk1
gene_id = 'EB52861' # Foxn4

black_fig()
sc.pl.umap(
    fn_adata,
    color=gene_id,
    # Setting a smaller point size to prevent overlap
    size=10,
    add_outline=False,
    layer="log1p",
    save = f'_{gene_id}_umap_fontes_neurales_only.pdf'
)

sc.pl.scatter(
    fn_adata,
    x='x_adjusted',
    y='y_adjusted',
    color=gene_id,
    size=4,
    layers="log1p",
    save=f"_{gene_id}_spatial_fontes_neurales_only.pdf"
)

#%% Violin plot of gene expression faceted by 'Tissue' and 'position' using plotnine
gene_ids = ['EB45560', #Ascl1-1
            'EB46007', #Ascl1-2
            'EB29593', # Lhx3
            'EB17597', #Runx1   
            'EB23886', #Pax6
]

#%% get dataframe for plotnine that includes expression of genes in gene_ids
df_list = []
for gene_id in gene_ids:
    gene_name = fn_adata.var.loc[gene_id, 'unique_gene_name']
    expr_values = fn_adata[:, gene_id].layers['log1p'].toarray().flatten()
    df_temp = pd.DataFrame({
        'expression': expr_values,
        'Tissue': fn_adata.obs['Tissue'].values,
        'position': fn_adata.obs['position'].values,
        'gene_id': gene_id,
        'gene_name': gene_name
    })
    df_list.append(df_temp)
df = pd.concat(df_list, ignore_index=True)

#%%
white_fig()
p = (
    pn.ggplot(df, pn.aes( y='expression', fill='Tissue'))
    #+ pn.geom_violin(scale='width', adjust=1.0)
    + pn.geom_boxplot(width=0.4, outlier_size=0.5)
    + pn.facet_grid('gene_name ~ position',scales='free')
    + pn.scale_y_continuous(limits=(0, None))
    + pn.scale_fill_brewer(type='qual', palette='Set2')
    + pn.scale_color_brewer(type='qual', palette='Set2')
    + pn.theme_bw()
    + pn.theme(
        figure_size=(8, 8),
        axis_text_x=element_text(rotation=45, hjust=1),
        strip_text=element_text(size=10)
    )
    + pn.ggtitle("Gene Expression in Fontes Neurales Cells by Position")
)
p.show()

#%%
p.save(f"{figdir}/fontes_neurales_gene_expression_boxplot.pdf", width=8, height=8)

#####################
# Up close visuals
#####################
# %%
pretty_sections = [#'no_chase_3',
                   #'04day_2',
                   #'07day_1',
                   #'07day_2',
                   #'14day_7',
                   '14day_6',
]

#%%
pretty_adata = adata[adata.obs['section_name'].isin(pretty_sections)].copy()

#%%
# Set the mean of x_adjusted and y_adjusted for each section to zero
for section in pretty_sections:
    mean_x = pretty_adata.obs.loc[pretty_adata.obs['section_name'] == section, 'x_adjusted'].mean()
    pretty_adata.obs.loc[pretty_adata.obs['section_name'] == section, 'x_adjusted'] -= mean_x
    mean_y = pretty_adata.obs.loc[pretty_adata.obs['section_name'] == section, 'y_adjusted'].mean()
    pretty_adata.obs.loc[pretty_adata.obs['section_name'] == section, 'y_adjusted'] -= mean_y 

# Shift the two sections apart on x axis for better visualization
shift_amount = 12000  # Adjust this value as needed for spacing
for _ in range(len(pretty_sections)):
    section = pretty_sections[_]
    pretty_adata.obs.loc[pretty_adata.obs['section_name'] == section, 'x_adjusted'] += _ * shift_amount 

#%%
black_fig()
sc.pl.scatter(
    pretty_adata,
    x='x_adjusted',
    y='y_adjusted',
    color="fontes_neurales",
    size=25,
    frameon=False,
    palette=['#555555','#377eb8',"#e21d1d","#9e3387"],
    save=f"_{section}_Fontes_Neurales_spatial.pdf",
)

#%%
target_gene = 'EB31550'  
sc.pl.scatter(
    pretty_adata,
    x='x_adjusted',
    y='y_adjusted',
    color=target_gene,
    size=25,
    frameon=False,
    palette='viridis',
    layers='log1p',
    #save=f"_{section}_{target_gene}_Tissue_spatial.pdf",
)

sc.pl.umap(
    pretty_adata,
    color=target_gene,
    size=20,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    #save = f'_{section}_{target_gene}_umap.pdf'
)

#%%
def plot_pretty_sections(adata,gene_id = 'EB45560',scale=True, layer=None, scalebar_size_um=100,**kwargs):
    ps = []
    x_lim = (adata.obs['x_adjusted'].min(), adata.obs['x_adjusted'].max())
    y_lim = (adata.obs['y_adjusted'].min(), adata.obs['y_adjusted'].max())
    p = plot_gene(adata,
            gene_id=gene_id,
            x='x_adjusted',
            y='y_adjusted',
            alpha=1,
            use_gene_name=True,
            layer=layer,
            **kwargs
            )
    p += theme(
        axis_text_x=element_blank(),  # Removes x-axis tick labels
        axis_text_y=element_blank(),  # Removes y-axis tick labels
        axis_title_x=element_blank(), # Removes x-axis title
        axis_title_y=element_blank(), # Removes y-axis title
        axis_ticks_x=element_blank(), # Removes x-axis ticks
        axis_ticks_y=element_blank(), # Removes y-axis ticks
        panel_grid_major_x=element_blank(), # Removes major x-gridlines
        panel_grid_minor_x=element_blank(), # Removes minor x-gridlines
        panel_grid_major_y=element_blank(), # Removes major y-gridlines
        panel_grid_minor_y=element_blank(), # Removes minor y-gridlines
        axis_line_x=element_blank(), # Removes x-axis line
        axis_line_y=element_blank()  # Removes y-axis line
    )
    
    for layer in add_scale_bar(x_lim, y_lim, color='white', size_um=scalebar_size_um):
        p = p + layer

    return p

#%% target genes
target_genes = ['EB45560', #Ascl1-1
                'EB46007', #Ascl1-2
                'EB17597', #Runx1
                'EB08075', #Ngn1
                'EB22657', #Sox2
                'EB23886', #Pax6
                'EB10890', #Nkx2-5
                'EB17694', #Nkx2-1
                ]
plots = []

for gene in target_genes:
    print(f"Plotting {gene}...")
    p = plot_pretty_sections(pretty_adata, gene_id = gene, layer='log1p') 
    p += theme(figure_size=(12, 10))
    plots.append(p)
    p.save(f"{figdir}/pretty_sections/{gene}_pretty_sections.svg", format='svg')

#%% Additionally mentioned genes
#%%
mentioned_genes = {
    'EB31550':'vvl',
    'EB38164': 'pang1',
    'EB01471': 'ems',
    'EB04221': 'gsc',
}

other_gene_plots = []

for gene_id, gene_name in mentioned_genes.items():
    print(f"Plotting {gene_id} ({gene_name})...")
    p = plot_pretty_sections(adata, gene_id = gene_id, layer='log1p',scalebar_size_um=500) 
    p += theme(figure_size=(12, 10))
    other_gene_plots.append(p)
    p.save(f"{figdir}/gene_callouts/{gene_id}_{gene_name}_pretty_sections.png", format='png')


#%%
##################################
# f-ara-EdU analysis
###################################

#%% EdU sections to visualize
edu_sections = ['no_chase_1',
                'no_chase_2',
                #'no_chase_3',
                #   '07day_1',
                '14day_7',
                '14day_5',
                #'14day_1',
                #'14day_2',
                #'14day_3',
                #'14day_4',
                '14day_6',
]

#%% Create edu subset adata
edu_adata = adata[adata.obs['section_name'].isin(edu_sections)].copy()

#%% log-transform mean_intensity_edu in new column log1p_edu
edu_adata.obs['log1p_edu'] = np.log1p(edu_adata.obs['mean_intensity_edu'])

#%% Plut log1p_edu in umap
black_fig()

#%% Edu distribution in no-chase
sc.pl.umap(
    edu_adata[edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2']), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    save = f'_EdU_no_chase.pdf',
    title="f-ara-EdU mean intensity - 0d chase"
)

#%% Edu distribution in 14d chase sections
sc.pl.umap(
    edu_adata[~edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2']), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    save = f'_EdU_14d_chase.pdf',
    title="f-ara-EdU mean intensity - 14d chase"
)

#%% EdU in no chase sections in cells annotated as 'Brain'
sc.pl.umap(
    edu_adata[(edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2'])) & (edu_adata.obs['Tissue']=='Brain'), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    title="f-ara-EdU Intensity after 0d chase\nfor cells annotated as within 'Brain'",
    save = f'_EdU_no_chase_in_brain.pdf'
)

#%% EdU in no chase sections in cells annotated as 'Not Brain'
sc.pl.umap(
    edu_adata[(edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2'])) & (edu_adata.obs['Tissue']!='Brain'), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    title="f-ara-EdU Intensity after 0d chase\nfor cells annotated as 'Not Brain'",
    save = f'_EdU_no_chase_not_in_brain.pdf'
)

#%% EdU in 14d chase sections in cells annotated as 'Brain'
sc.pl.umap(
    edu_adata[(~edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2'])) & (edu_adata.obs['Tissue']=='Brain'), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    title="f-ara-EdU Intensity after 14d chase\nfor cells annotated as within 'Brain'",
    save = f'_EdU_14d_chase_in_brain.pdf'
)

#%% EdU in 14d chase sections in cells annotated as 'Not Brain'
sc.pl.umap(
    edu_adata[(~edu_adata.obs['section_name'].isin(['no_chase_1', 'no_chase_2'])) & (edu_adata.obs['Tissue']!='Brain'), :],
    color="mean_intensity_edu",
    size=8,
    frameon=False,
    cmap='viridis',
    add_outline=False,
    legend_fontsize="medium",
    title="f-ara-EdU mean pixel Intensity after 14d chase\nfor cells annotated as 'Not Brain'",
    save = f'_EdU_14d_chase_not_in_brain.pdf'
)

#%%
black_fig()

for section in edu_sections:
    # sc.pl.scatter(
    #     edu_adata[edu_adata.obs['section_name'] == section, :],
    #     x='x_adjusted',
    #     y='y_adjusted',
    #     color="mean_intensity_edu",
    #     size=20,
    #     frameon=False,
    #     palette='viridis',
    #     save=f"_EdU_{section}_mean_edu.pdf",
    # )
    p = plot_pretty_sections(
        edu_adata[edu_adata.obs['section_name'] == section, :],
        gene_id='mean_intensity_edu',
        layer=None,
        scalebar_size_um=500,
    )
    p.save(f"{figdir}/EdU_spatial/{section}_mean_edu_pretty_sections.pdf", width=12, height=10)
    #p.show()

#%%
# black_fig()

# for section in edu_sections:
#     sc.pl.scatter(
#         edu_adata[edu_adata.obs['section_name'] == section, :],
#         x='x_adjusted',
#         y='y_adjusted',
#         color="log1p_edu",
#         size=20,
#         frameon=False,
#         palette='viridis',
#         #save=f"_{section}_log1p_edu.pdf",
#     )

# %% Fox genes
# fox_genes = [ 'EB02432',
#                 'EB06288',
#                 'EB10087',
#                 'EB15728',
#                 'EB15939',
#                 'EB16006',
#                 'EB21496',
#                 'EB24830',
#                 'EB36069',
#                 'EB37598',
#                 'EB39014',
#                 'EB48951',
#                 'EB52861',
# ]

# for gene in fox_genes:
#     (plot_pretty_sections(gene_id = gene) + theme(figure_size=(15, 12))).save(f"{figdir}/{gene}_pretty_sections.pdf")
# %%

#########################
# Glia
#########################
#%% Glial annotation
edu_adata.obs['neural_cell_type'] = 'Other'

#%% todo: pick correct leiden clusters for glia vs neurons
edu_adata.obs.loc[edu_adata.obs['leiden_res_1.0'].isin(['11']), 'neural_cell_type'] = 'Glia'
edu_adata.obs.loc[edu_adata.obs['leiden_res_1.0'].isin(['2','3','4','5','8','9','10','12','13','14','15','16','17','18','19','20','21','22']), 'neural_cell_type'] = 'Neuron'

#%% change order of categorical values
edu_adata.obs['neural_cell_type'] = pd.Categorical(
    edu_adata.obs['neural_cell_type'],
    categories=['Neuron', 'Glia', 'Other'],
    ordered=True
)

#%%
black_fig()
sc.pl.umap(
    edu_adata,
    color="neural_cell_type",
    size=8,
    frameon=False,
    palette=['#377eb8', '#e41a1c', '#cccccc'],
    add_outline=True,
    legend_fontsize="medium",
    save = f'_neural_cell_type_umap.pdf')

##################
# Updated Fig 5 UMAP
##################
#%% Fig5 annotation
edu_adata.obs['fig5_cell_type'] = 'Other'

#%% todo: pick correct leiden clusters for glia vs neurons
edu_adata.obs.loc[edu_adata.obs['leiden_res_1.0'].isin(['11']), 'fig5_cell_type'] = 'Glia'
edu_adata.obs.loc[edu_adata.obs['leiden_res_1.0'].isin(['2','3','4','5','8','9','10','12','13','14','15','16','17','18','19','20','21','22']), 'fig5_cell_type'] = 'Neuron'
edu_adata.obs.loc[edu_adata.obs['Tissue'].isin(['Fontes Neurales']), 'fig5_cell_type'] = 'Fontes_Neurales'

#%% change order of categorical values 
#
edu_adata.obs['fig5_cell_type'] = pd.Categorical(
    edu_adata.obs['fig5_cell_type'],
    categories=['Neuron', 'Glia', 'Fontes_Neurales','Other'],
    ordered=True
)

#%%
black_fig()
sc.pl.umap(
    edu_adata,
    color="fig5_cell_type",
    size=30,
    frameon=False,
    palette=['#377eb8', '#e41a1c', "#f9f918", "#46464611"],
    #palette=['#000000'],
    add_outline=False,
    #outline_color=('black','white'),
    #outline_width=(0.25, 0.1),
    legend_fontsize="medium",
    save = f'_fig5_cell_type_umap.pdf')


# %%   plotnine density plot of log1p_edu intensity across neural cell types
white_fig()

p = (
    pn.ggplot(edu_adata.obs[edu_adata.obs['fig5_cell_type'] != 'Other'], pn.aes(x='mean_intensity_edu', fill='fig5_cell_type')) 
    + pn.geom_histogram(alpha=0.75, position='identity', bins=100)
    #+ theme_black_minimal()
    + pn.ggtitle("f-ara-EdU Intensity Distribution\nacross Neural Cell Types")
    + pn.scale_fill_manual(values=['#377eb8', '#e41a1c',"#f9f918", '#cccccc'])
    + pn.xlab("log1p(f-ara-EdU Intensity)")
    + pn.scale_x_log10()
    #+ pn.scale_y_log10()
    + pn.ylab("Cell Counts")
    + pn.facet_wrap('chase_incubation_days')
)

p.show()

p.save(f"{figdir}/EdU_intensity_histogram_neural_cell_types.pdf", width=5, height=4)

# %%   plotnine density plot of log1p_edu intensity across neural cell types
white_fig()

p = (
    pn.ggplot(edu_adata.obs[edu_adata.obs['fig5_cell_type'] != 'Other'], pn.aes(x='mean_intensity_edu', fill='fig5_cell_type')) 
    + pn.geom_histogram(alpha=0.75, position='identity', bins=100)
    #+ theme_black_minimal()
    + pn.ggtitle("f-ara-EdU Intensity Distribution\nacross Neural Cell Types")
    + pn.scale_fill_manual(values=['#377eb8', '#e41a1c',"#f9f918", '#cccccc'])
    + pn.scale_y_log10()
    + pn.xlab("log1p(f-ara-EdU Intensity)")
    + pn.ylab("Cell Counts")
    + pn.scale_x_log10()
    + pn.facet_wrap('chase_incubation_days')
    + pn.facet_grid("Brain Structure ~ chase_incubation_days")

)

p.show()

p.save(f"{figdir}/EdU_intensity_histogram_neural_cell_types_faceted.pdf", width=6, height=4)

# %%
p = (
    pn.ggplot(edu_adata.obs[edu_adata.obs['fig5_cell_type'] != 'Other'], pn.aes(x='log1p_edu', fill='fig5_cell_type')) 
    + pn.geom_density(alpha=0.5)
    #+ theme_black_minimal()
    + pn.ggtitle("f-ara-EdU Intensity Distribution\nacross Neural Cell Types")
    + pn.scale_fill_manual(values=['#377eb8', '#e41a1c',"#f9f918", '#cccccc'])
    + pn.xlab("log1p(f-ara-EdU Intensity)")
    + pn.ylab("Density")
    + pn.facet_grid("Brain Structure ~ chase_incubation_days")
)

p.show()

p.save(f"{figdir}/EdU_intensity_density_neural_cell_types.pdf", width=6, height=4)

#%%#################################
# Additional supplemental figures
####################################

# Figure S4: Spot annotation plots
target_obs = ['leiden_res_1.0', 'position', 'Brain Structure', 'Optic Lobe', 'White Body', 'fontes_neurales','sample', 'log1p_total_counts', 'death_age_days']

n_rows = len(target_obs)
black_fig()
fig, axes = plt.subplots(n_rows, 2, figsize=(24, 10 * n_rows))

panel_labels = list(string.ascii_uppercase)

for i, x in enumerate(target_obs):
    n_axes_before = len(fig.axes)
    sc.pl.scatter(
        adata,
        x='x_adjusted',
        y='y_adjusted',
        color=x,
        size=3,
        frameon=False,
        show=False,
        ax=axes[i, 0],
        legend_loc='none',
    )
    
    axes[i, 0].set_xlabel('')
    axes[i, 0].set_ylabel('')
    axes[i, 0].set_xticks([])
    axes[i, 0].set_yticks([])
    # Remove colorbar added by scatter
    while len(fig.axes) > n_axes_before:
        fig.axes[-1].remove()
        
    sc.pl.umap(
        adata,
        color=x,
        size=3,
        frameon=False,
        show=False,
        ax=axes[i, 1],
        colorbar_loc='right',
        legend_loc='best',  # Disable the giant colorbar
    )
    
    # Add panel labels
    axes[i, 0].text(-0.05, 1.05, panel_labels[i], transform=axes[i, 0].transAxes,
                    fontsize=50, fontweight='bold', va='bottom', ha='right')

# Remove all colorbars from the figure
# for ax in fig.axes:
#     if ax not in axes.flatten():
#         ax.remove()

#%%
plt.tight_layout()
fig.savefig(f'{figdir}/Figure_S7_SOLAR_spot_annotation_plots.png', bbox_inches='tight',dpi=400)

#%% Same plot but for each leiden_res_1.0 cluster separately
# n_clusters = adata.obs['leiden_res_1.0'].nunique()
# black_fig()
# fig, axes = plt.subplots(n_clusters, 2, figsize=(24, 10 * n_clusters))
# panel_labels = list(string.ascii_uppercase) + [f'A{x}' for x in string.ascii_uppercase]
#
# for i, cluster in enumerate(adata.obs['leiden_res_1.0'].cat.categories):
#     n_axes_before = len(fig.axes)
#     sc.pl.scatter(
#         adata,
#         x='x_adjusted',
#         y='y_adjusted',
#         color='leiden_res_1.0',
#         groups=cluster,
#         size=5,
#         frameon=False,
#         show=False,
#         ax=axes[i, 0],
#         legend_loc='none',
#     )
#    
#     axes[i, 0].set_xlabel('')
#     axes[i, 0].set_ylabel('')
#     axes[i, 0].set_xticks([])
#     axes[i, 0].set_yticks([])
#     # Remove colorbar added by scatter
#     while len(fig.axes) > n_axes_before:
#         fig.axes[-1].remove()
#        
#     sc.pl.umap(
#         adata,
#         color='leiden_res_1.0',
#         groups=cluster,
#         size=5,
#         frameon=False,
#         show=False,
#         ax=axes[i, 1],
#         colorbar_loc='right',
#         legend_loc='best',  # Disable the giant colorbar
#     )
#    
#     # Add panel labels
#     axes[i, 0].text(-0.05, 1.05, panel_labels[i], transform=axes[i, 0].transAxes,
#                     fontsize=50, fontweight='bold', va='bottom', ha='right')
#
# plt.tight_layout()
# fig.savefig(f'{figdir}/Figure_S4_2_SOLAR_by_leiden_1.0_cluster.png', bbox_inches='tight')


#%%#########################
# Save final anndatas
############################
wb_adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_wb_subset.h5ad')

#%%
fn_adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_fn_subset.h5ad')

# %%
edu_adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_edu_subset.h5ad')

# %%
adata.write(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_resubmission.h5ad')

# %% The following can only be executed after the pseudotime analysis has been performed and 'results/white_body_pseudotime_values.csv' exists
# Load pseudotime values and add to adata
pseudotime_df = pd.read_csv('results_for_resubmission/fontes_neurales_pseudotime_values.csv', index_col=0)
adata.obs['t'] = pseudotime_df['t']

# %%
# plot_pretty_sections with gene_id='t' to visualize pseudotime values in spatial context
pseudotime_plot = plot_pretty_sections(adata, 
                                    gene_id='t',
                                    layer=None,
                                    scalebar_size_um=500,
                                    size=0.01,
                                    ) + pn.scale_color_cmap('viridis', na_value="#151515") + pn.ggtitle("Pseudotime values in Fontes Neurales cells") 

pseudotime_plot.show()
pseudotime_plot.save(f"{figdir}/fontes_neurales_pseudotime_spatial_plot_with_scalebar.pdf", width=20, height=20)

# %%
