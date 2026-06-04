#%% imports
import scanpy as sc 
import statsmodels.api as sm
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import IPython.display
from matplotlib_inline.backend_inline import set_matplotlib_formats
IPython.display.set_matplotlib_formats = set_matplotlib_formats
import scipy.stats 
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import dask.array as da
import plotnine as pn
import patsy

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

#%% load data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20260513_loyal_annotations_and_figures_for_manuscript_post_pearson_resubmission.h5ad')
#%%
if isinstance(adata.layers['counts'], da.Array):
    print("  Converting from Dask to numpy/sparse...")
    counts = adata.layers['counts'].compute()
else:
    counts = adata.layers['counts']

#%%
# Try to compute if it has the method
if hasattr(counts, 'compute'):
    print("  Calling .compute() on Dask array...")
    counts = counts.compute()

# If it's sparse, convert to dense
if sparse.issparse(counts):
    print("  Converting sparse to dense...")
    counts = counts.toarray()
else:
    print("  Converting to numpy array...")
    counts = np.array(counts)
    
#%% prepare data for regression
#adata.obs["is_white_body"] = adata.obs["Tissue"]=='White Body'

#%%
# white_body = pd.Categorical(adata.obs['is_white_body'])
# predictor = white_body.codes
# X = sm.add_constant(predictor)

#%%
gene_names = np.array(adata.var['gene_name'].values)

#%%
design_formula = '~ fontes_neurales + ID'  # or '~ condition + section'
X = patsy.dmatrix(design_formula, data=adata.obs, return_type='dataframe')

#%% Normalize by n_counts
log_ncounts = np.log1p(adata.obs['n_counts'].values)

#%% estimate gene-wise dispersions for negbinomial model
from statsmodels.nonparametric.smoothers_lowess import lowess

def estimateDispersions(
    adata,
    layer="counts",
    design_cols=None,
    X=None,
    offset_key="log_ncounts",
    size_factor_key=None,
    shrink=True,
    lowess_frac=0.3,
    min_alpha=1e-8,
    max_alpha=100.0,
    inplace=True,
    var_prefix="nb_",
    verbose_every=500,
):
    """
    Approximate per-gene NB2 dispersions (alpha) from AnnData.
    Var(Y) = mu + alpha * mu^2

    Returns:
        pd.DataFrame indexed by adata.var_names with columns:
        alpha_raw, alpha_shrunk, mu_mean, fit_ok
    """
    # 1) counts matrix
    Y = adata.layers[layer] if layer is not None else adata.X
    if Y is None:
        raise ValueError(f"No matrix found for layer={layer!r}.")
    n_cells, n_genes = Y.shape

    if sparse.issparse(Y):
        Y = Y.tocsc()  # fast per-gene column slicing

    # 2) design matrix
    if X is None:
        if design_cols is None or len(design_cols) == 0:
            X_df = pd.DataFrame({"Intercept": np.ones(n_cells)}, index=adata.obs_names)
        else:
            d = adata.obs[design_cols].copy()
            X_df = pd.get_dummies(d, drop_first=True, dtype=float)
            X_df.insert(0, "Intercept", 1.0)
    else:
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X_arr = np.asarray(X, dtype=float)
            if X_arr.ndim != 2:
                raise ValueError("X must be 2D.")
            X_df = pd.DataFrame(
                X_arr,
                index=adata.obs_names,
                columns=[f"term_{i}" for i in range(X_arr.shape[1])],
            )
        if X_df.shape[0] != n_cells:
            raise ValueError("X rows must equal adata.n_obs.")

    # 3) offset
    if offset_key is not None and offset_key in adata.obs.columns:
        offset = np.asarray(adata.obs[offset_key], dtype=float)
    else:
        if size_factor_key is not None and size_factor_key in adata.obs.columns:
            lib = np.asarray(adata.obs[size_factor_key], dtype=float)
        else:
            lib = np.asarray(Y.sum(axis=1)).ravel() if sparse.issparse(Y) else Y.sum(axis=1)
        offset = np.log(np.clip(lib, 1.0, None))

    alpha_raw = np.full(n_genes, np.nan, dtype=float)
    mu_mean = np.full(n_genes, np.nan, dtype=float)
    fit_ok = np.zeros(n_genes, dtype=bool)

    # 4) gene-wise MOM dispersion from Poisson fit
    for i in range(n_genes):
        y = Y[:, i].toarray().ravel().astype(float) if sparse.issparse(Y) else np.asarray(Y[:, i], dtype=float).ravel()
        if y.sum() <= 0:
            continue
        try:
            fit = sm.GLM(y, X_df, family=sm.families.Poisson(), offset=offset).fit()
            mu = np.clip(np.asarray(fit.fittedvalues, dtype=float), 1e-8, None)
            den = np.sum(mu * mu)
            if den <= 0:
                continue
            a = np.sum((y - mu) ** 2 - y) / den
            alpha_raw[i] = np.clip(a, min_alpha, max_alpha)
            mu_mean[i] = mu.mean()
            fit_ok[i] = True
        except Exception:
            continue

        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"  dispersion fit: {i+1}/{n_genes}")

    # 5) optional LOWESS shrinkage on mean-dispersion trend
    alpha_shrunk = alpha_raw.copy()
    if shrink:
        m = fit_ok & np.isfinite(alpha_raw) & np.isfinite(mu_mean) & (mu_mean > 0)
        if m.sum() >= 10:
            lx = np.log(mu_mean[m])
            ly = np.log(np.clip(alpha_raw[m], min_alpha, max_alpha))
            ly_hat = lowess(ly, lx, frac=lowess_frac, return_sorted=False)
            alpha_shrunk[m] = np.exp(ly_hat)

    alpha_shrunk = np.clip(alpha_shrunk, min_alpha, max_alpha)

    out = pd.DataFrame(
        {
            "alpha_raw": alpha_raw,
            "alpha_shrunk": alpha_shrunk,
            "mu_mean": mu_mean,
            "fit_ok": fit_ok,
        },
        index=adata.var_names,
    )

    if inplace:
        adata.var[f"{var_prefix}alpha_raw"] = out["alpha_raw"].values
        adata.var[f"{var_prefix}alpha"] = out["alpha_shrunk"].values
        adata.var[f"{var_prefix}mu_mean"] = out["mu_mean"].values
        adata.var[f"{var_prefix}fit_ok"] = out["fit_ok"].values

    return out

#%% run estimate dispersions
disp = estimateDispersions(
      adata,
      layer="counts",
      #offset_key="log_ncounts",     # or set None to auto-compute from library size
      shrink=True,
  )


#%%
results_list = []
# Fit GLM for each gene
for i in range(counts.shape[1]):
    gene_name = gene_names[i]
    gene_id = adata.var_names[i]
    alpha = adata.var.loc[gene_id, 'nb_alpha'] if 'nb_alpha' in adata.var.columns else None

    print(gene_name)
    
    # Simple indexing - counts is now definitely a regular numpy array
    y = counts[:, i]
    
    try:
        model = sm.GLM(y, 
                       X, 
                       family=sm.families.NegativeBinomial(alpha=alpha),
                       offset=log_ncounts)
        
        result = model.fit()
        
        results_list.append({
            'gene': gene_name,
            'gene_id': gene_id,
            'coef': result.params[1],
            'pval': result.pvalues[1],
            'stderr': result.bse[1],
            'log2fc': result.params[1] / np.log(2),
        })
    except Exception as e:
        results_list.append({
            'gene': gene_name,
            'gene_id': gene_id,
            'coef': np.nan,
            'pval': np.nan,
            'stderr': np.nan,
            'log2fc': np.nan,
        })
    
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(gene_names)} genes...")

#%%
#################
# Visualize
#################
# Create results dataframe
results_df = pd.DataFrame(results_list)

# Multiple testing correction
results_df['padj'] = multipletests(results_df['pval'].fillna(1), method='fdr_bh')[1]

# Add unique_gene_name by matching on gene_id
results_df = results_df.merge(adata.var[['unique_gene_name']], left_on='gene_id',  right_index=True, how='left')

#%%
# Sort by significance
results_df = results_df.sort_values('padj')

print(results_df.head(20))

# Prepare data
plot_data = results_df.copy()
plot_data['neg_log10_pval'] = -np.log10(plot_data['padj'])

# Handle inf values
max_log_p = plot_data['neg_log10_pval'].replace([np.inf], np.nan).max()
plot_data['neg_log10_pval'] = plot_data['neg_log10_pval'].replace([np.inf, -np.inf], max_log_p * 1.1)

#%%
# Significance categories
alpha = 0.0001
plot_data['significance'] = 'Not Significant'
plot_data.loc[(plot_data['padj'] < alpha) & (plot_data['coef'] > 0), 'significance'] = 'Up'
plot_data.loc[(plot_data['padj'] < alpha) & (plot_data['coef'] < 0), 'significance'] = 'Down'

# Get up and down regulated genes
up_genes = plot_data[(plot_data['padj'] < alpha) & (plot_data['coef'] > 0)]
down_genes = plot_data[(plot_data['padj'] < alpha) & (plot_data['coef'] < 0)]
all_sig_genes = plot_data[plot_data['padj'] < alpha]
# Write up_genes to .csv
up_genes.to_csv(f'{resultsdir}/up_genes.csv', index=False)
all_sig_genes.to_csv(f'{resultsdir}/all_significant_genes_fontes_neurales_vs_other_tissue.csv', index=False)

# Label top n up and top n down
n=10
plot_data['label'] = ''

# Get top n upregulated (positive coef, lowest padj)
up_genes_subset = plot_data[(plot_data['padj'] < alpha) & (plot_data['coef'] > 0)].nsmallest(n, 'padj')
plot_data.loc[up_genes_subset.index, 'label'] = up_genes_subset['gene']

# Get top n downregulated (negative coef, lowest padj)
down_genes_subset = plot_data[(plot_data['padj'] < alpha) & (plot_data['coef'] < 0)].nsmallest(n, 'padj')
plot_data.loc[down_genes_subset.index, 'label'] = down_genes_subset['gene']

# %% Create plot
volcano = (
    pn.ggplot(plot_data, pn.aes(x='coef', y='neg_log10_pval')) +
    pn.geom_point(pn.aes(color='significance'), alpha=0.6, size=2) +
    pn.geom_text(
        pn.aes(label='label'),
        size=8,
        nudge_y=0.5,
        data=plot_data[plot_data['label'] != '']
    ) +
    pn.geom_hline(yintercept=-np.log10(alpha), linetype='dashed', color='red', alpha=0.5) +
    pn.geom_vline(xintercept=0, linetype='dashed', color='gray', alpha=0.5) +
    pn.scale_color_manual(
        values={
            'Not Significant': '#CCCCCC',
            'Up': '#E41A1C',
            'Down': '#377EB8'
        }
                          ) +
    pn.labs(
        title='Gene Expression ~ 1 + Fontes Neurales vs Other Tissues',
        x='Beta Coefficient',
        y='-Log10(Adjusted P-value)',
        color='Significance'
    ) +
    pn.theme_bw() +
    pn.theme(
        figure_size=(7,6),
        legend_position='right',
        plot_title=pn.element_text(size=14, weight='bold')
    )
)

#%%
volcano.show()

#%% Save plot
volcano.save(f'{figdir}/volcano_plot_fontes_neurales_vs_other_tissue.pdf', width=7, height=6)

#%% Create a matrix of average expression in each tissue for all genes
tissue_agg = sc.get.aggregate(adata, 
                              by='Tissue', 
                              layer='counts', 
                              func='mean')

#tissue_agg = np.log10(tissue_agg)

# Create a DataFrame for heatmap
colnames = tissue_agg.obs['Tissue'].values
rownames = tissue_agg.var['unique_gene_name'].values
tissue_agg_df = pd.DataFrame(data=tissue_agg.layers['mean'].T, index=rownames, columns=colnames)

#%% Using tissue_agg_df, create a heatmap of significant genes
alpha = 1e-300
sig_genes = results_df[results_df['padj'] < alpha]['unique_gene_name'].values
up_sig_genes = results_df[(results_df['padj'] < alpha) & (results_df['coef'] > 0)]['unique_gene_name'].values

heatmap_data = tissue_agg_df.loc[up_sig_genes]

#%% Draw heatmap with seaborn
heatmap = sns.clustermap(
    z_score="row",
    data=heatmap_data,
    cmap='bwr',
    #standard_scale=0,
    figsize=(8, 24),
    center=0,
    yticklabels=1,
)

#%%
#heatmap.show()

#%%
heatmap.savefig(f'{figdir}/heatmap_fontes_neurales_vs_other_tissue_siggenes.pdf', bbox_inches='tight')

#%% 
heatmap_data.shape

# %%
# Print which genes are labeled
print(f"\nTop {n} Upregulated Genes:")
print(up_genes[['gene', 'log2fc', 'padj']])

print(f"\nTop {n} Downregulated Genes:")
print(down_genes[['gene', 'log2fc', 'padj']])
# %%
