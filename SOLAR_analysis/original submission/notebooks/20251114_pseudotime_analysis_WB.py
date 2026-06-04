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

seed = 1191979
#%%
def white_fig(figsize=[12,12]):
    sc.set_figure_params(facecolor='white',
                         format='pdf', 
                         frameon=False, 
                         dpi=300, 
                         figsize=figsize,
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

#black_fig()

#%% load WB data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20251106_loyal_annotations_and_figures_for_manuscript_post_pearson_wb_subset.h5ad')

#%% Remove cluster of 'hematopoietic' WB cells
# Using wb_leiden_res_1.0 clustering
clusters_to_remove = ['14','16']

adata = adata[~adata.obs['wb_leiden_res_1.0'].isin(clusters_to_remove)].copy()
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
scf.pl.trajectory(adata,
                  basis="pca",
                  arrows=True,
                  arrow_offset=1,
                  color_cells="t",
                  cmap="viridis",
                  size=20,
                  save="_White_Body_pseudotime_trajectory.pdf"
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
                  save=f"_White_Body_cluster_{c}_trends.pdf"
                  )

# %%
black_fig()
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="t",
    size=3,
    palette="viridis",
    save="_White_Body_only_pseudotime.pdf"
    
)

# %%
white_fig()
scf.pl.trends(adata,
                  basis="pca",
                  plot_emb=False,
                  #return_genes=True,
                  #feature_cmap="viridis",
                  save="_White_Body_cluster_trends.pdf",
                  )

gene_order = scf.pl.trends(adata,
                  basis="pca",
                  plot_emb=False,
                  return_genes=True,
                  #feature_cmap="viridis",
                  #save="_White_Body_cluster_trends.pdf",
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
              save="_White_Body_pseudotime_ordered_matrix.pdf",
              #gene_symbols = 'unique_gene_name'
              )


# %%
# List adata.var['unique_gene_name'] in order of gene_order
ordered_gene_names = adata.var['unique_gene_name'].loc[gene_order].values
# %%
