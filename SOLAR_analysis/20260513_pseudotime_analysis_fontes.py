#%% imports
import scanpy as sc 
import statsmodels.api as sm
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from scipy import sparse
import scFates as scf
import IPython.display
from matplotlib_inline.backend_inline import set_matplotlib_formats
IPython.display.set_matplotlib_formats = set_matplotlib_formats
from plot_utils import white_fig, black_fig

seed = 1191979

#%%
sc.set_figure_params(vector_friendly=False)
sc.settings.verbosity = 3
figdir = "figures_for_resubmission"
resultsdir = "results_for_resubmission"
sc.settings.figdir = figdir + "/"
plt.rcParams["axes.grid"] = False

from plotnine import theme, element_rect, element_text, element_line, element_blank
pd.set_option('mode.copy_on_write', True)  # This might help with the view issue

plt.rcParams['pdf.fonttype'] = 42


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

#black_fig()

#%% load WB data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_fn_subset.h5ad')

# %%
sc.pp.filter_genes(adata,min_cells=5)
#adata.X = adata.layers['log1p'].copy()
adata.X = adata.layers['counts']

#%%
#sc.pp.scale(adata)

#%%
# sc.pp.highly_variable_genes(adata,min_disp=0.0)
sc.experimental.pp.highly_variable_genes(adata, flavor="pearson_residuals",n_top_genes=500)

#%% find high variance pearson residual genes
sc.experimental.pp.normalize_pearson_residuals(adata)

#%% PCA
#sc.pp.scale(adata)
sc.tl.pca(adata,
          n_comps=50,
          random_state=seed,
          use_highly_variable = True,
          svd_solver='arpack',
          zero_center=True
          )

sc.pl.pca_variance_ratio(adata, log=False)
#%%
# Pseudotime analysis on whole body data
# sc.pl.umap(adata,
#           color=['EB08075'], # Ngn1
#           #dimensions=[(0,1),(1,2)],
#           )

#%%
sc.pl.pca(adata,
        color=['EB49399'], # Mcm6
        dimensions=[(0,1),(1,2)],
        size=25
)

# %%
scf.tl.curve(adata,
             Nodes=6,
             use_rep="pca",
             ndims_rep=2,)

# %%
scf.pl.graph(adata,basis="pca")

# %%
sc.pl.pca(sc.AnnData(adata.obsm["X_R"],obsm=adata.obsm),color="2",cmap="Reds")

# %%
scf.tl.root(adata,"EB49399")

# %%
scf.tl.pseudotime(adata,n_jobs=8,n_map=100,seed=42)

# %%
sc.pl.pca(adata,color="t")

#%%
ax = scf.pl.trajectory(adata,
                  basis="pca",
                  arrows=True,
                  arrow_offset=1,
                  color_cells="t",
                  cmap="viridis",
                  size=20,
                  save="_Fontes_Neurales_pseudotime_trajectory.pdf",
                  )


# %%
sc.pl.pca(adata,color="milestones")

# %%
start = adata.uns['graph']['root']
end = adata.uns['graph']['tips'][adata.uns['graph']['tips']!=start][0]
scf.tl.rename_milestones(adata,new={str(start):"Proliferating Progenitors",str(end): "Maturing Neurons"})

# %%
sc.pl.umap(adata,color="milestones")

# %%
scf.pl.milestones(adata,basis="pca",annotate=True)

# %%
scf.tl.linearity_deviation(adata,
                           start_milestone="Proliferating Progenitors",
                           end_milestone="Maturing Neurons",
                           n_jobs=20,plot=True,basis="pca")

# %%
scf.pl.linearity_deviation(adata,
                           start_milestone="Proliferating Progenitors",
                           end_milestone="Maturing Neurons")

# %%
test_genes = ["EB22391","EB00351","EB17597",'EB14502']
sc.pl.pca(adata,color=test_genes,cmap="RdBu_r")

# %%
scf.tl.test_association(adata,n_jobs=20)

#%%
scf.pl.test_association(adata)

# %%
scf.tl.test_association(adata,reapply_filters=True,A_cut=0.5)
scf.pl.test_association(adata)

#%%
scf.tl.fit(adata,n_jobs=20)

#%%
for gene in test_genes:
    scf.pl.single_trend(adata,gene,basis="pca",color_exp="k")

#%%
scf.tl.cluster(adata,n_neighbors=50,metric="correlation")

#%%
adata.var.clusters.unique()

#%%
white_fig(figsize=[6,4])

for c in adata.var["clusters"].unique():
    scf.pl.trends(adata,
                  features=adata.var_names[adata.var.clusters==c],
                  basis="pca",
                  save=f"_Fontes_Neurales_cluster_{c}_trends.pdf"
                  )

# %%
black_fig()
ax = sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="t",
    size=3,
    palette="viridis",
    #save="_Fontes_Neurales_only_pseudotime.pdf"
    show=False,
)

add_scale_bar_mpl(ax, um_per_pixel=0.216, color='white', size_um=100)
plt.show()

#%%
plt.savefig(f'{figdir}/fontes_neurales_pseudotime_spatial_plot_with_scalebar.pdf', bbox_inches='tight')

# %%
white_fig()
scf.pl.trends(adata,
                  basis="pca",
                  plot_emb=False,
                  #return_genes=True,
                  #feature_cmap="viridis",
                  save="_Fontes_Neurales_cluster_trends.pdf",
                  )

gene_order = scf.pl.trends(adata,
                  basis="pca",
                  plot_emb=False,
                  return_genes=True,
                  #feature_cmap="viridis",
                  #save="_Fontes_Neurales_cluster_trends.pdf",
                  )

# %% matrix plot ordered by progression along pseudotime

#%%
#adata.var_names = adata.var['unique_gene_name']
gene_order_fixed = adata.var['unique_gene_name'].loc[gene_order].values
bdata = adata.copy()
bdata.var_names = bdata.var['unique_gene_name'].values

#%%
scf.pl.matrix(bdata,
              features = gene_order_fixed,
              nbins=50,
              cmap="RdBu_r",
              annot_top = True,
              save="_Fontes_Neurales_pseudotime_ordered_matrix.pdf",
              #gene_symbols = 'unique_gene_name'
              )


# %%
# List adata.var['unique_gene_name'] in order of gene_order
ordered_gene_names = adata.var['unique_gene_name'].loc[gene_order].values
# %% Write significant genes to csv
adata.var.to_csv(f'{resultsdir}/fontes_neurales_pseudotime_siggenes.csv')
# %% export 't' pseudotime values and roilabel to csv
adata.obs[['t']].to_csv(f'{resultsdir}/fontes_neurales_pseudotime_values.csv')

#%%