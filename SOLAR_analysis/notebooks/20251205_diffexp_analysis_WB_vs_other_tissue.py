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
import plotnine as pn
import seaborn as sns
import patsy

#%%
#plt.rcParams['pdf.fonttype'] = 42

#%% load data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20251106_loyal_annotations_and_figures_for_manuscript_post_pearson.h5ad')

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
adata.obs["is_white_body"] = adata.obs["Tissue"]=='White Body'

#%%
# white_body = pd.Categorical(adata.obs['is_white_body'])
# predictor = white_body.codes
# X = sm.add_constant(predictor)

#%%
gene_names = np.array(adata.var['gene_name'].values)

#%%
design_formula = '~ is_white_body + ID'  # or '~ condition + section'
X = patsy.dmatrix(design_formula, data=adata.obs, return_type='dataframe')

#%% Normalize by n_counts
log_ncounts = np.log1p(adata.obs['n_counts'].values)

#%%
results_list = []
# Fit GLM for each gene
for i in range(counts.shape[1]):
    gene_name = gene_names[i]
    print(gene_name)
    
    # Simple indexing - counts is now definitely a regular numpy array
    y = counts[:, i]
    
    try:
        model = sm.GLM(y, 
                       X, 
                       family=sm.families.NegativeBinomial(),
                       offset=log_ncounts)
        
        result = model.fit()
        
        results_list.append({
            'gene': gene_name,
            'coef': result.params[1],
            'pval': result.pvalues[1],
            'stderr': result.bse[1],
            'log2fc': result.params[1] / np.log(2),
        })
    except Exception as e:
        results_list.append({
            'gene': gene_name,
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
up_genes.to_csv('results/up_genes.csv', index=False)
all_sig_genes.to_csv('results/all_significant_genes_white_body_vs_other_tissue.csv', index=False)

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
        title='Gene Expression ~ 1 + White Body Anterior vs Posterior',
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
volcano.save('figures/volcano_plot_white_body_vs_other_tissue.pdf', width=7, height=6)

#%% Create a matrix of average expression in each tissue for all genes
tissue_agg = sc.get.aggregate(adata, 
                              by='Tissue', 
                              layer='counts', 
                              func='mean')

#tissue_agg = np.log10(tissue_agg)

# Create a DataFrame for heatmap
colnames = tissue_agg.obs['Tissue'].values
rownames = tissue_agg.var['gene_name'].values
tissue_agg_df = pd.DataFrame(data=tissue_agg.layers['mean'].T, index=rownames, columns=colnames)

#%% Using tissue_agg_df, create a heatmap of significant genes
alpha = 1e-300
sig_genes = results_df[results_df['padj'] < alpha]['gene'].values
up_sig_genes = results_df[(results_df['padj'] < alpha) & (results_df['coef'] > 0)]['gene'].values

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
#plt.show()

#%%
plt.savefig('figures/heatmap_white_body_vs_other_tissue_siggenes.pdf', bbox_inches='tight')

#%% 
heatmap_data.shape

# %%
# Print which genes are labeled
print(f"\nTop {n} Upregulated Genes:")
print(up_genes[['gene', 'log2fc', 'padj']])

print(f"\nTop {n} Downregulated Genes:")
print(down_genes[['gene', 'log2fc', 'padj']])
# %%
