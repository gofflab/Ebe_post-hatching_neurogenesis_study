#! /usr/bin/env python3
# mamba activate edu_quant

#%% Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import plotnine as pn
from scipy import stats
import matplotlib.pyplot as plt

# At the start of your script
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'

#%% Load dataset
data = pd.read_csv('20251205_resection_quant.csv')

data

#%% Remove 'potential_outlier' = True rows
data = data[data['potential_artifact'] == False].copy()

#%% Convert 'edu_cell_count' column to (edu_cell_count/area)*1000000 and make a float
scaling_factor = 1e6 #(in mm**2)
scaling_factor = 1e5  #(in 100000 um**2)
data['normalized_count'] = (data['edu_cell_count'] / data['area']) * scaling_factor

# create an 'animal' column by combining 'transplant_date' and 'embryo'
data['animal'] = data['date'].astype(str) + ' ' + data['embryo'].astype(str)

# Set levels for categorical variables
data['relative_side'] = pd.Categorical(data['relative_side'], categories=['ipsilateral', 'contralateral'], ordered=True)

#%% Summary statistics
summary_by_type = data.groupby(['resection', 'tissue', 'relative_side'])['normalized_count'].agg(
    ['mean', 'std', 'sem', 'count']
).reset_index()
print("\nMean normalized counts by resection type, tissue, and side:")
print(summary_by_type.to_string())

#%% ============================================================================
# STATISTICAL TESTS - Mixed Effects Model accounting for animal variance
# ============================================================================

print("\n" + "=" * 70)
print("STATISTICAL ANALYSIS OF RESECTION EdU+ CELL COUNTS")
print("Mixed Effects Models (accounting for animal-to-animal variance)")
print("=" * 70)

# Store results for plotting
hypothesis_results = []

#%% -----------------------------------------------------------------------------
# For each resection type and tissue, test ipsi vs contra using mixed model
# Model: normalized_count ~ relative_side + (1|animal)
# -----------------------------------------------------------------------------

print("\n" + "-" * 70)
print("TEST: Side comparison (ipsi vs contra) - Mixed Effects Model")
print("Model: normalized_count ~ relative_side, random intercept for animal")
print("-" * 70)

for resection in data['resection'].unique():
    for tissue in data['tissue'].unique():
        subset = data[(data['resection'] == resection) & (data['tissue'] == tissue)].copy()
        
        # Fit mixed effects model with animal as random effect
        # Testing effect of relative_side on normalized_count
        model = smf.mixedlm("normalized_count ~ C(relative_side, Treatment('ipsilateral'))", 
                           data=subset, 
                           groups=subset['animal'])
        result = model.fit(reml=True)
        
        # Extract the coefficient and p-value for contralateral effect
        # (This is the difference from ipsilateral baseline)
        coef = result.fe_params["C(relative_side, Treatment('ipsilateral'))[T.contralateral]"]
        pval = result.pvalues["C(relative_side, Treatment('ipsilateral'))[T.contralateral]"]
        se = result.bse["C(relative_side, Treatment('ipsilateral'))[T.contralateral]"]
        
        # Get means for each side
        ipsi_mean = subset[subset['relative_side'] == 'ipsilateral']['normalized_count'].mean()
        contra_mean = subset[subset['relative_side'] == 'contralateral']['normalized_count'].mean()
        ipsi_sem = subset[subset['relative_side'] == 'ipsilateral']['normalized_count'].sem()
        contra_sem = subset[subset['relative_side'] == 'contralateral']['normalized_count'].sem()
        n_ipsi = len(subset[subset['relative_side'] == 'ipsilateral'])
        n_contra = len(subset[subset['relative_side'] == 'contralateral'])
        n_animals = subset['animal'].nunique()
        
        print(f"\n{resection} Resection - {tissue}:")
        print(f"  N animals: {n_animals}")
        print(f"  Ipsilateral:    n={n_ipsi}, mean={ipsi_mean:.2f} ± {ipsi_sem:.2f}")
        print(f"  Contralateral:  n={n_contra}, mean={contra_mean:.2f} ± {contra_sem:.2f}")
        print(f"  Difference (contra - ipsi): {coef:.2f} ± {se:.2f}")
        print(f"  Mixed model p-value: {pval:.6f}")
        
        # Also show the random effect variance
        print(f"  Animal variance: {result.cov_re.iloc[0,0]:.2f}")
        print(f"  Residual variance: {result.scale:.2f}")
        
        hypothesis_results.append({
            'resection': resection,
            'tissue': tissue,
            'comparison': 'ipsi_vs_contra',
            'coefficient': coef,
            'std_error': se,
            'p_value': pval,
            'mean_ipsi': ipsi_mean,
            'mean_contra': contra_mean,
            'n_animals': n_animals
        })

#%% ============================================================================
# ALTERNATIVE: Paired analysis by animal (aggregate slices first)
# ============================================================================

print("\n" + "=" * 70)
print("ALTERNATIVE: Paired t-test on animal means")
print("(Aggregating slices within each animal first)")
print("=" * 70)

for resection in data['resection'].unique():
    for tissue in data['tissue'].unique():
        subset = data[(data['resection'] == resection) & (data['tissue'] == tissue)].copy()
        
        # Aggregate by animal and side
        animal_means = subset.groupby(['animal', 'relative_side'])['normalized_count'].mean().reset_index()
        animal_means_wide = animal_means.pivot(index='animal', columns='relative_side', values='normalized_count')
        
        # Paired t-test
        ipsi_vals = animal_means_wide['ipsilateral'].values
        contra_vals = animal_means_wide['contralateral'].values
        
        t_stat, p_val = stats.ttest_rel(ipsi_vals, contra_vals)
        
        print(f"\n{resection} Resection - {tissue}:")
        print(f"  N animals (paired): {len(ipsi_vals)}")
        print(f"  Mean ipsilateral:   {ipsi_vals.mean():.2f}")
        print(f"  Mean contralateral: {contra_vals.mean():.2f}")
        print(f"  Mean difference:    {(contra_vals - ipsi_vals).mean():.2f}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.6f}")

#%% ============================================================================
# PLOTTING WITH SIGNIFICANCE BRACKETS
# ============================================================================

norm_boxplots_with_stats = []

for resection in data['resection'].unique():
    subset = data[data['resection'] == resection].copy()
    
    # Create a combined group variable for x-axis
    subset['group'] = subset['tissue'] + '\n' + subset['relative_side'].astype(str)
    
    group_order = [
        'Central Brain\nipsilateral',
        'Central Brain\ncontralateral', 
        'Optic Lobe\nipsilateral',
        'Optic Lobe\ncontralateral'
    ]
    subset['group'] = pd.Categorical(subset['group'], categories=group_order, ordered=True)
    
    # Calculate overall max for bracket positioning
    overall_max = subset['normalized_count'].max()
    
    # --- Get p-values from mixed model results ---
    p_cb = None
    p_ol = None
    for result in hypothesis_results:
        if result['resection'] == resection:
            if result['tissue'] == 'Central Brain':
                p_cb = result['p_value']
            elif result['tissue'] == 'Optic Lobe':
                p_ol = result['p_value']
    
    # Print for verification
    print(f"\n{'='*60}")
    print(f"{resection} Resection - P-values for plot (from mixed model)")
    print(f"{'='*60}")
    print(f"CB ipsi vs CB contra: p = {p_cb:.6f}")
    print(f"OL ipsi vs OL contra: p = {p_ol:.6f}")
    
    # --- Format p-values ---
    def format_pval(p):
        if p is None:
            return ""
        elif p < 0.0001:
            return "p < 0.0001"
        elif p < 0.001:
            return f"p = {p:.4f}"
        else:
            return f"p = {p:.3f}"
    
    # --- Create bracket dataframes ---
    # Bracket heights
    y1 = overall_max * 1.05   # Within CB bracket
    y2 = overall_max * 1.05   # Within OL bracket  
    
    text_offset = overall_max * 0.03
    
    # Using numeric x positions: 1=CB ipsi, 2=CB contra, 3=OL ipsi, 4=OL contra
    bracket_data = pd.DataFrame([
        # CB ipsi vs CB contra (positions 1 to 2)
        {'x': 1, 'xend': 2, 'y': y1, 'yend': y1},
        # OL ipsi vs OL contra (positions 3 to 4)
        {'x': 3, 'xend': 4, 'y': y2, 'yend': y2},
    ])
    
    text_data = pd.DataFrame([
        {'x': 1.5, 'y': y1 + text_offset, 'label': format_pval(p_cb)},
        {'x': 3.5, 'y': y2 + text_offset, 'label': format_pval(p_ol)},
    ])
    
    # Create the plot
    p = (pn.ggplot(subset, pn.aes(x='group', y='normalized_count', fill='tissue'))
         + pn.geom_boxplot(alpha=0.8, outlier_shape='')
         + pn.geom_jitter(
            pn.aes(shape='animal'),
            width=0.15,
            size=2,
            alpha=0.7,
            color='black',
        )
         + pn.theme_minimal()
         + pn.labs(
             title=f'Area Normalized F-ara-EdU\u207A Cell Counts for {resection} Resection',
             x='',
             y='Normalized F-ara-EdU\u207A Cell Count\n(cells/ 100,000 µm²)'
         )
         + pn.scale_fill_brewer(type='qual', palette='Set1')
         + pn.theme(
             legend_position='top',
             text=pn.element_text(family='DejaVu Sans'),
             figure_size=(8, 5),
             axis_text_x=pn.element_text(size=9)
         )
         # Add bracket lines
         + pn.geom_segment(
             pn.aes(x='x', xend='xend', y='y', yend='yend'),
             data=bracket_data,
             inherit_aes=False,
             size=0.5,
             color='black'
         )
         # Add p-value text
         + pn.geom_text(
             pn.aes(x='x', y='y', label='label'),
             data=text_data,
             inherit_aes=False,
             size=8,
             ha='center',
             va='bottom'
         )
    )
    
    norm_boxplots_with_stats.append(p)
    p.show()

#%% Save plots
width = 8
height = 5
for i, resection in enumerate(data['resection'].unique()):
    norm_boxplots_with_stats[i] = norm_boxplots_with_stats[i] + pn.theme(figure_size=(width, height))
    norm_boxplots_with_stats[i].save(f"Figure_6_resection_quant_{resection}.pdf")

#%% ============================================================================
# SUMMARY TABLE
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY OF STATISTICAL TESTS (Mixed Effects Model)")
print("=" * 70)

results_df = pd.DataFrame(hypothesis_results)
print(results_df.to_string())

#%%