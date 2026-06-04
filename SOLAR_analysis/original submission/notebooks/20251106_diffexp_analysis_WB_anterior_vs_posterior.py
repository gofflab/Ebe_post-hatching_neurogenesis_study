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

#%%
#plt.rcParams['pdf.fonttype'] = 42

#%% load data from h4ad
datadir = 'cleaned/'

adata = sc.read_h5ad(datadir + 'combined/20251106_loyal_annotations_and_figures_for_manuscript_post_pearson_wb_subset.h5ad')

#%% Remove cells with 'White Body' not in ['white_body_anterior', 'white_body_posterior']
adata = adata[adata.obs['White Body'].isin(['white_body_anterior', 'white_body_posterior'])].copy()


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
white_body = pd.Categorical(adata.obs['White Body'])
predictor = white_body.codes

#%%
gene_names = np.array(adata.var['gene_name'].values)

# Add intercept
X = sm.add_constant(predictor)

#%%
results_list = []
# Fit GLM for each gene
for i in range(counts.shape[1]):
    gene_name = gene_names[i]
    print(gene_name)
    
    # Simple indexing - counts is now definitely a regular numpy array
    y = counts[:, i]
    
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
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
all_significant_genes.to_csv('results/significant_genes_white_body_anterior_vs_posterior.csv', index=False)

#%%
# Get top n upregulated (positive coef, lowest padj)
up_genes = plot_data[(plot_data['padj'] < 0.05) & (plot_data['coef'] > 0)].nsmallest(n, 'padj')
plot_data.loc[up_genes.index, 'label'] = up_genes['gene']

# Get top n downregulated (negative coef, lowest padj)
down_genes = plot_data[(plot_data['padj'] < 0.05) & (plot_data['coef'] < 0)].nsmallest(n, 'padj')
plot_data.loc[down_genes.index, 'label'] = down_genes['gene']

# Create plot
volcano = (
    ggplot(plot_data, aes(x='coef', y='neg_log10_pval')) +
    geom_point(aes(color='significance'), alpha=0.6, size=2) +
    geom_text(
        aes(label='label'),
        size=8,
        nudge_y=0.5,
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
        title='Gene Expression ~ 1 + White Body Anterior vs Posterior',
        x='Beta Coefficient',
        y='-Log10(Adjusted P-value)',
        color='Significance'
    ) +
    theme_bw() +
    theme(
        figure_size=(7,6),
        legend_position='right',
        plot_title=element_text(size=14, weight='bold')
    )
)

#%%
volcano.show()

#%% Save plot
volcano.save('figures/volcano_plot_y_afo_white_body_anterior_vs_posterior_wb_only.pdf', width=7, height=6)

# %%
# Print which genes are labeled
print("\nTop 5 Upregulated Genes:")
print(up_genes[['gene', 'log2fc', 'padj']])

print("\nTop 5 Downregulated Genes:")
print(down_genes[['gene', 'log2fc', 'padj']])
# %%
