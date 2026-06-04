#%% imports
import scanpy as sc 
import statsmodels.api as sm
import anndata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats 
from scipy import sparse
from statsmodels.stats.multitest import multipletests
import dask.array as da
import IPython.display
from matplotlib_inline.backend_inline import set_matplotlib_formats
IPython.display.set_matplotlib_formats = set_matplotlib_formats
from plot_utils import white_fig, black_fig

#%%
sc.set_figure_params(vector_friendly=False)
sc.settings.verbosity = 3
figdir = "figures_for_resubmission"
resultsdir = "results_for_resubmission"
sc.settings.figdir = figdir + "/"
plt.rcParams["axes.grid"] = False

from plotnine import theme, element_rect, element_text, element_line, element_blank, annotate, arrow, scale_color_manual, labs, theme_bw
pd.set_option('mode.copy_on_write', True)  # This might help with the view issue

plt.rcParams['pdf.fonttype'] = 42

#%% load data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20251106_loyal_annotations_and_figures_for_manuscript_post_pearson_wb_subset.h5ad')

#%% Remove cells with 'White Body' not in ['white_body_anterior', 'white_body_posterior']
adata = adata[adata.obs['White Body'].isin(['white_body_anterior', 'white_body_posterior'])].copy()

#%% Remove cluster of 'hematopoietic' WB cells
# Using wb_leiden_res_1.0 clustering
clusters_to_remove = ['14','16']

adata = adata[~adata.obs['wb_leiden_res_1.0'].isin(clusters_to_remove)].copy()

#%%
if isinstance(adata.layers['counts'], da.Array):
    print("  Converting from Dask to numpy/sparse...")
    counts = adata.layers['counts'].compute()
else:
    counts = adata.layers['counts']

#%% Rename 'White Body' categories to 'Fontes Neurales'
adata.obs['White Body'] = adata.obs['White Body'].replace({
    'white_body_anterior': 'fontes_neurales_anterior',
    'white_body_posterior': 'fontes_neurales_posterior'
})

# rename 'White Body' column to 'Fontes Neurales'
adata.obs.rename(columns={'White Body': 'Fontes Neurales'}, inplace=True)

#%% Reorder categories to have 'fontes_neurales_anterior' as numerator
adata.obs['Fontes Neurales'] = pd.Categorical(
    adata.obs['Fontes Neurales'],
    categories=['fontes_neurales_posterior','fontes_neurales_anterior'],
    ordered=True
)

#%% Test scatter
black_fig()
sc.pl.scatter(
    adata,
    x='x_adjusted',
    y='y_adjusted',
    color="Fontes Neurales",
    size=2
)

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
fontes_neurales = pd.Categorical(adata.obs['Fontes Neurales'])
predictor = fontes_neurales.codes

#%%
gene_names = np.array(adata.var['gene_name'].values)

#%% Normalize by n_counts
log_ncounts = np.log1p(adata.obs['n_counts'].values)

# Add intercept
X = sm.add_constant(predictor)

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

#%%
# Sort by significance
results_df = results_df.sort_values('padj')

print(results_df.head(20))

# %%
from plotnine import *
import pandas as pd
import numpy as np

# Prepare data
plot_data = results_df.copy()
plot_data['neg_log10_pval'] = -np.log10(plot_data['padj'])

# Handle inf values
max_log_p = plot_data['neg_log10_pval'].replace([np.inf], np.nan).max()
plot_data['neg_log10_pval'] = plot_data['neg_log10_pval'].replace([np.inf, -np.inf], max_log_p * 1.1)

# Significance categories
plot_data['significance'] = 'Not Significant'
plot_data.loc[(plot_data['padj'] < 0.05) & (plot_data['coef'] > 0), 'significance'] = 'Up'
plot_data.loc[(plot_data['padj'] < 0.05) & (plot_data['coef'] < 0), 'significance'] = 'Down'

# Label top n up and top n down
n=10
plot_data['label'] = ''

#%% write significant genes to csv
alpha = 0.05
all_significant_genes = plot_data[plot_data['padj'] < alpha]
all_significant_genes.to_csv(f'{resultsdir}/significant_genes_fontes_neurales_anterior_vs_posterior.csv', index=False)

#%%
# Get top n upregulated (positive coef, lowest padj)
up_genes = plot_data[(plot_data['padj'] < 0.05) & (plot_data['coef'] > 0)].nsmallest(n, 'padj')
plot_data.loc[up_genes.index, 'label'] = up_genes['gene']

# Get top n downregulated (negative coef, lowest padj)
down_genes = plot_data[(plot_data['padj'] < 0.05) & (plot_data['coef'] < 0)].nsmallest(n, 'padj')
plot_data.loc[down_genes.index, 'label'] = down_genes['gene']

#%%
# Create plot
white_fig()
volcano = (
    ggplot(plot_data, aes(x='coef', y='neg_log10_pval')) +
    geom_point(aes(color='significance'), alpha=0.6, size=2) +
    geom_text(
        aes(label='label'),
        size=8,
        #nudge_y=0.5,
        # adjust_text={
        #  # Change 'expand_points' to 'expand'
        #  "expand": (1.2, 1.5), 
        #  "arrowprops": {"arrowstyle": "-", "color": "gray"}
        # },
        data=plot_data[plot_data['label'] != '']
    ) +
    geom_hline(yintercept=-np.log10(0.05), linetype='dashed', color='red', alpha=0.5) +
    geom_vline(xintercept=0, linetype='dashed', color='gray', alpha=0.5) +
    scale_color_manual(values={
        'Not Significant': '#CCCCCC',
        'Up': '#E41A1C',
        'Down': '#377EB8'
    }) +
    labs(
        title='Gene Expression ~ 1 + Fontes Neurales Anterior vs Posterior',
        x='Beta Coefficient',
        y='-Log10(Adjusted P-value)',
        color='Significance'
    ) +
    theme_bw() +
    scale_y_log10() +
    theme(
        figure_size=(7,6),
        legend_position='right',
        plot_title=element_text(size=14, weight='bold')
    ) +
    annotate('segment',
         x=1.5, xend=2.5,
         y=-np.log10(0.05), yend=-np.log10(0.05),
         arrow=arrow(length=0.2), color='black') +
annotate('text',
         x=2.0, y=-np.log10(0.05) * 1.3,
         label='Anterior > Posterior', size=10) +
annotate('segment',
         x=-1.5, xend=-2.5,
         y=-np.log10(0.05), yend=-np.log10(0.05),
         arrow=arrow(length=0.2), color='black') +
annotate('text',
         x=-2.0, y=-np.log10(0.05) * 1.3,
         label='Posterior > Anterior', size=10)
)

#%%
volcano.show()

#%% Save plot
volcano.save(f'{figdir}/volcano_plot_y_afo_fontes_neurales_anterior_vs_posterior_fn_only.pdf', width=7, height=6)

# %%
# Print which genes are labeled
print(f"\nTop {n} Anterior-biased Genes:")
print(up_genes[['gene', 'log2fc', 'padj']])

print(f"\nTop {n} Posterior-biased Genes:")
print(down_genes[['gene', 'log2fc', 'padj']])
# %%
