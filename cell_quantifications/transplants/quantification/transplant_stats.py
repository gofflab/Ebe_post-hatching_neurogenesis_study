#! /usr/bin/env python3
# mamba activate edu_quant

#%% Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices, bs
import plotnine as pn
from scipy import stats
import matplotlib.pyplot as plt

# At the start of your script
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
plt.rcParams['font.family'] = 'sans-serif'

#%% Load dataset
data = pd.read_csv('20251201_transplant_quant.csv')

data

#%% Convert 'edu_cell_count' column to (edu_cell_count/area)*1000000 and make a float
scaling_factor = 1e6 #(in mm**2)
scaling_factor = 1e5  #(in 100000 um**2)
data['normalized_count'] = (data['edu_cell_count'] / data['area']) * scaling_factor

# create an 'animal' column by combining 'transplant_date' and 'embryo'
data['animal'] = data['transplant_date'].astype(str) + ' ' + data['embryo'].astype(str)

#%%
#data = data.dropna()

# Set levels for categorical variables
data['relative_side'] = pd.Categorical(data['relative_side'], categories=['ipsilateral', 'contralateral'], ordered=True)

# %% Remap transplant_type with unicode right arrows replacing the '-'
data['transplant_type'] = data['transplant_type'].str.replace('-', '\u2192')

#%%
# Summary statistics
summary_by_type = data.groupby(['transplant_type', 'tissue'])['normalized_count'].agg(['mean', 'sem', 'count']).reset_index()
print("\nMean normalized counts by transplant type and tissue:")
print(summary_by_type)


# %% Create a faceted boxplot of raw cell counts
raw_boxplot = (pn.ggplot(data, pn.aes(x='animal', y='edu_cell_count', fill='tissue'))
     + pn.geom_boxplot(position=pn.position_dodge(0.9))
     + pn.facet_grid('relative_side~tissue+transplant_type',scales='free_x')
     + pn.theme_bw()
     + pn.labs(title='Transplant EdU\u207A Cell Counts by Tissue',
                y='Raw F-ara-EdU\u207A Cell Count')
     + pn.scale_fill_brewer(type='qual', palette='Set1')
     + pn.theme(legend_position='top', 
        legend_box='horizontal',
        axis_text_x=pn.element_text(rotation=-90, hjust=1))
    )

raw_boxplot.show()

# %% Subset data by transplant_type and create boxplots for each faceted in the same way
#data['animal_side'] = data['animal'].astype(str) + '_' + data['relative_side'].astype(str)

for transplant in data['transplant_type'].unique():
    subset = data[data['transplant_type'] == transplant]
    p = (pn.ggplot(subset, pn.aes(x='tissue', y='edu_cell_count', fill='tissue',alpha='relative_side'))
         + pn.geom_boxplot(position=pn.position_dodge(0.95))
         + pn.geom_jitter(
            pn.aes(
                   shape='animal', 
                   group='relative_side',
                   ),  # Map animal to color
            position=pn.position_jitterdodge(jitter_width=0.075, dodge_width=0.95),
            size=1.5,
            alpha=0.6,
            color='black',
            inherit_aes=True,  # Don't inherit fill and alpha from main aes
            data=subset  # Explicitly pass data
        )
         + pn.scale_alpha_manual(values={'ipsilateral': 1.0, 'contralateral': 0.5})
         + pn.facet_grid('~tissue',scales='free_x')
         + pn.theme_minimal()
         + pn.labs(title=f'Raw F-ara-EdU\u207A Cell Counts for {transplant} Transplant',
                    y='Raw F-ara-EdU\u207A Cell Count')
         + pn.scale_fill_brewer(type='qual', palette='Set1')
         + pn.theme(legend_position='top', 
            legend_box='horizontal',
            text=pn.element_text(family='DejaVu Sans')
            #axis_text_x=pn.element_text(rotation=-45, hjust=1)
            )
        )
    p.show()

# %% Create faceted boxplot of normalized cell counts
normalized_boxplot = (pn.ggplot(data, pn.aes(x='animal', y='normalized_count', fill='tissue'))
     + pn.geom_boxplot(position=pn.position_dodge(0.9))
     + pn.facet_grid('relative_side~tissue+transplant_type',scales='free_x')
     + pn.theme_bw()
     + pn.labs(title='Transplant EdU\u207A Normalized Cell Counts by Tissue',
                y='Normalized EdU\u207A Cell Count (per 100,000 µm²)')
     + pn.scale_fill_brewer(type='qual', palette='Set1')
     + pn.theme(legend_position='top', 
        legend_box='horizontal',
        axis_text_x=pn.element_text(rotation=-90, hjust=1))
     
    )

normalized_boxplot.show()

# %% Subset data by transplant_type and create boxplots for each faceted in the same way
#data['animal_side'] = data['animal'].astype(str) + '_' + data['relative_side'].astype(str)
norm_boxplots = []

for transplant in data['transplant_type'].unique():
    subset = data[data['transplant_type'] == transplant]
    p = (pn.ggplot(subset, pn.aes(x='tissue', y='normalized_count', fill='tissue',alpha='relative_side'))
         + pn.geom_boxplot(position=pn.position_dodge(0.8))
         + pn.geom_jitter(
            pn.aes(
                   shape='animal', 
                   group='relative_side',
                   ),  # Map animal to color
            position=pn.position_jitterdodge(jitter_width=0.2, dodge_width=0.8),
            size=1.5,
            alpha=0.6,
            color='black',
            inherit_aes=True,  # Don't inherit fill and alpha from main aes
            data=subset  # Explicitly pass data
        )
         + pn.scale_alpha_manual(values={'ipsilateral': 1.0, 'contralateral': 0.5})
         + pn.facet_grid('~tissue',scales='free_x')
         + pn.theme_minimal()
         + pn.labs(title=f'Area Normalized F-ara-EdU\u207A Cell Counts for {transplant} Transplant',
                    y='Normalized F-ara-EdU\u207A Cell Count \n (cells/ 100,000 µm²)')
         + pn.scale_fill_brewer(type='qual', palette='Set1')
         + pn.theme(legend_position='top', 
            legend_box='horizontal',
            text=pn.element_text(family='DejaVu Sans')
            #axis_text_x=pn.element_text(rotation=-45, hjust=1)
            )
        )
    norm_boxplots.append(p)
    p.show()
    
#%%
width = 5
height= 4
for i,transplant in enumerate(["Anterior-Anterior","Posterior-Posterior"]):
    norm_boxplots[i] = norm_boxplots[i] + pn.theme(figure_size=(width, height))
    norm_boxplots[i].save(f"Figure_7_transplant_quant_{transplant}.pdf")

# %% Fit a mixed effects model with a random effect for animal
model_formula = 'normalized_count ~ C(tissue) * C(relative_side) * C(transplant_type)'
md = sm.MixedLM.from_formula(model_formula,
                             groups='animal', 
                             data=data)  # Fit separately for each tissue
mdf = md.fit()
print(mdf.summary())

#%%
#########################

# %%
# Hypothesis 1A: Anterior-Anterior preferentially populates Central Brain  (IPSILATERAL)
# Test: Do Anterior-Anterior transplants have higher counts in Central Brain than Optic Lobe?

print("\n" + "="*70)
print("Hypothesis 1A: Anterior-Anterior transplant preferentially populates Central Brain (IPSILATERAL)")
print("="*70)

# Filter for Anterior-Anterior transplants, ipsilateral only
ant_ant_ipsi = data[(data['transplant_type'] == 'Anterior\u2192Anterior') & 
                     (data['relative_side'] == 'ipsilateral')].copy()

# Check if we have data for both tissues
tissues_present = ant_ant_ipsi['tissue'].unique()
print(f"\nTissues present in Anterior\u2192Anterior ipsilateral: {tissues_present}")

if len(tissues_present) < 2:
    print(f"⚠️ Warning: Only {len(tissues_present)} tissue type(s) present. Cannot compare.")
else:
    # Compare Central Brain vs Optic Lobe
    cb_ant_ant = ant_ant_ipsi[ant_ant_ipsi['tissue'] == 'Central Brain']['normalized_count']
    ol_ant_ant = ant_ant_ipsi[ant_ant_ipsi['tissue'] == 'Optic Lobe']['normalized_count']
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Central Brain: mean = {cb_ant_ant.mean():.2f} ± {cb_ant_ant.sem():.2f} (SD = {cb_ant_ant.std():.2f}), n = {len(cb_ant_ant)}")
    print(f"  Optic Lobe:    mean = {ol_ant_ant.mean():.2f} ± {ol_ant_ant.sem():.2f} (SD = {ol_ant_ant.std():.2f}), n = {len(ol_ant_ant)}")
    print(f"  Difference:    {cb_ant_ant.mean() - ol_ant_ant.mean():.2f} cells/100,000 µm²")
    print(f"  Ratio (CB/OL): {cb_ant_ant.mean() / (ol_ant_ant.mean() + 1e-6):.2f}x")
    
    # Independent samples t-test
    t_stat_ant, p_val_ant = stats.ttest_ind(cb_ant_ant, ol_ant_ant)
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(((len(cb_ant_ant)-1)*cb_ant_ant.std()**2 + (len(ol_ant_ant)-1)*ol_ant_ant.std()**2) / 
                         (len(cb_ant_ant) + len(ol_ant_ant) - 2))
    cohens_d = (cb_ant_ant.mean() - ol_ant_ant.mean()) / pooled_std
    
    # 95% Confidence Interval for the difference
    se_diff = np.sqrt(cb_ant_ant.sem()**2 + ol_ant_ant.sem()**2)
    ci_lower = (cb_ant_ant.mean() - ol_ant_ant.mean()) - 1.96 * se_diff
    ci_upper = (cb_ant_ant.mean() - ol_ant_ant.mean()) + 1.96 * se_diff
    
    print(f"\nStatistical Test:")
    print(f"  t-test: t({len(cb_ant_ant) + len(ol_ant_ant) - 2}) = {t_stat_ant:.3f}, p = {p_val_ant:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")
    print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val_ant < 0.05:
        if cb_ant_ant.mean() > ol_ant_ant.mean():
            print(f"  ✓ SIGNIFICANT: Anterior\u2192Anterior transplants preferentially populate Central Brain (ipsilateral)")
            print(f"    Central Brain has {cb_ant_ant.mean() / (ol_ant_ant.mean() + 1e-6):.1f}x more cells than Optic Lobe")
        else:
            print(f"  ✗ UNEXPECTED: Anterior\u2192Anterior transplants higher in Optic Lobe")
    else:
        print(f"  ✗ NOT SIGNIFICANT: No strong preference detected (p = {p_val_ant:.4f})")
        print(f"    May need more samples or there may be no true difference")

#%% Save p-values for plotting later
hypothesis_results = []
hypothesis_results.append({
    'hypothesis': 'H1A_Anterior_Anterior_CB_vs_OL',
    't_stat': t_stat_ant,
    'p_value': p_val_ant,
    'cohens_d': cohens_d,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper,
    'mean_CB': cb_ant_ant.mean(),
    'mean_OL': ol_ant_ant.mean(),
    'n_CB': len(cb_ant_ant),
    'n_OL': len(ol_ant_ant)
})


# %%
# HYPOTHESIS 1B: Posterior\u2192Posterior preferentially populates Optic Lobe (IPSILATERAL)
# Test: Do Posterior-Posterior transplants have higher counts in Optic Lobe than Central Brain?

print("\n" + "="*70)
print("HYPOTHESIS 1B: Posterior\u2192Posterior transplant preferentially populates Optic Lobe (Ipsilateral only)")
print("="*70)

# Filter for Posterior-Posterior transplants, ipsilateral only
post_post_ipsi = data[(data['transplant_type'] == 'Posterior\u2192Posterior') & 
                     (data['relative_side'] == 'ipsilateral')].copy()   
# Check if we have data for both tissues
tissues_present = post_post_ipsi['tissue'].unique()
print(f"\nTissues present in Posterior\u2192Posterior ipsilateral: {tissues_present}")
if len(tissues_present) < 2:
    print(f"⚠️ Warning: Only {len(tissues_present)} tissue type(s) present. Cannot compare.")
else:
    # Compare Optic Lobe vs Central Brain
    ol_post_post = post_post_ipsi[post_post_ipsi['tissue'] == 'Optic Lobe']['normalized_count']
    cb_post_post = post_post_ipsi[post_post_ipsi['tissue'] == 'Central Brain']['normalized_count']
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Optic Lobe:   mean = {ol_post_post.mean():.2f} ± {ol_post_post.sem():.2f} (SD = {ol_post_post.std():.2f}), n = {len(ol_post_post)}")
    print(f"  Central Brain: mean = {cb_post_post.mean():.2f} ± {cb_post_post.sem():.2f} (SD = {cb_post_post.std():.2f}), n = {len(cb_post_post)}")
    print(f"  Difference:    {ol_post_post.mean() - cb_post_post.mean():.2f} cells/100,000 µm²")
    print(f"  Ratio (OL/CB): {ol_post_post.mean() / (cb_post_post.mean() + 1e-6):.2f}x")
    
    # Independent samples t-test
    t_stat_post, p_val_post = stats.ttest_ind(ol_post_post, cb_post_post)
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(((len(ol_post_post)-1)*ol_post_post.std()**2 + (len(cb_post_post)-1)*cb_post_post.std()**2) / 
                         (len(ol_post_post) + len(cb_post_post) - 2))
    cohens_d = (ol_post_post.mean() - cb_post_post.mean()) / pooled_std
    
    # 95% Confidence Interval for the difference
    se_diff = np.sqrt(ol_post_post.sem()**2 + cb_post_post.sem()**2)
    ci_lower = (ol_post_post.mean() - cb_post_post.mean()) - 1.96 * se_diff
    ci_upper = (ol_post_post.mean() - cb_post_post.mean()) + 1.96 * se_diff
    
    print(f"\nStatistical Test:")
    print(f"  t-test: t({len(ol_post_post) + len(cb_post_post) - 2 }) = {t_stat_post:.3f}, p = {p_val_post:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")
    print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]") 
    print(f"\nInterpretation:")
    if p_val_post < 0.05:
        if ol_post_post.mean() > cb_post_post.mean():
            print(f"  ✓ SIGNIFICANT: Posterior\u2192Posterior transplants preferentially populate Optic Lobe (Ipsilateral Only)")
            print(f"    Optic Lobe has {ol_post_post.mean() / (cb_post_post.mean() + 1e-6):.1f}x more cells than Central Brain")
        else:
            print(f"  ✗ UNEXPECTED: Posterior\u2192Posterior transplants higher in Central Brain")
    else:
        print(f"  ✗ NOT SIGNIFICANT: No strong preference detected (p = {p_val_post:.4f})")
        print(f"    May need more samples or there may be no true difference")  

#%% Save p-values for plotting later
hypothesis_results.append({
    'hypothesis': 'H1B_Posterior_Posterior_OL_vs_CB',
    't_stat': t_stat_post,
    'p_value': p_val_post,
    'cohens_d': cohens_d,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper,
    'mean_OL': ol_post_post.mean(),
    'mean_CB': cb_post_post.mean(),
    'n_OL': len(ol_post_post),
    'n_CB': len(cb_post_post)
})

# %%
# HYPOTHESIS 2: Ipsilateral Restriction
# Test: Are transplanted cells restricted to the ipsilateral (same) side?

print("\n" + "="*70)
print("HYPOTHESIS 2: Ipsilateral Restriction")
print("="*70)

# Test for each transplant type separately
for transplant in data['transplant_type'].unique():
    print(f"\n{'='*70}")
    print(f"Testing: {transplant}")
    print(f"{'='*70}")
    
    # Filter for this transplant type
    transplant_data = data[data['transplant_type'] == transplant].copy()
    
    # Check if we have both sides
    sides_present = transplant_data['relative_side'].unique()
    print(f"\nSides present: {sides_present}")
    
    if len(sides_present) < 2:
        print(f"⚠️ Warning: Only {len(sides_present)} side(s) present. Cannot compare.")
        continue
    
    # Get ipsilateral and contralateral data
    ipsi = transplant_data[transplant_data['relative_side'] == 'ipsilateral']['normalized_count']
    contra = transplant_data[transplant_data['relative_side'] == 'contralateral']['normalized_count']
    
    # Descriptive statistics
    print(f"\nDescriptive Statistics:")
    print(f"  Ipsilateral:     mean = {ipsi.mean():.2f} ± {ipsi.sem():.2f} (SD = {ipsi.std():.2f}), n = {len(ipsi)}")
    print(f"  Contralateral:   mean = {contra.mean():.2f} ± {contra.sem():.2f} (SD = {contra.std():.2f}), n = {len(contra)}")
    print(f"  Difference:      {ipsi.mean() - contra.mean():.2f} cells/100,000 µm²")
    print(f"  Ratio (Ipsi/Contra): {ipsi.mean() / (contra.mean() + 1e-6):.2f}x")
    
    # Independent samples t-test
    t_stat, p_val = stats.ttest_ind(ipsi, contra)
    
    # Calculate Cohen's d effect size
    pooled_std = np.sqrt(((len(ipsi)-1)*ipsi.std()**2 + (len(contra)-1)*contra.std()**2) / 
                         (len(ipsi) + len(contra) - 2))
    cohens_d = (ipsi.mean() - contra.mean()) / pooled_std
    
    # 95% Confidence Interval for the difference
    se_diff = np.sqrt(ipsi.sem()**2 + contra.sem()**2)
    ci_lower = (ipsi.mean() - contra.mean()) - 1.96 * se_diff
    ci_upper = (ipsi.mean() - contra.mean()) + 1.96 * se_diff
    
    print(f"\nStatistical Test:")
    print(f"  t-test: t({len(ipsi) + len(contra) - 2}) = {t_stat:.3f}, p = {p_val:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f} ({'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'} effect)")
    print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")
    
    # Interpretation
    print(f"\nInterpretation:")
    if p_val < 0.05:
        if ipsi.mean() > contra.mean():
            print(f"  ✓ SIGNIFICANT: {transplant} cells are restricted to ipsilateral side")
            print(f"    Ipsilateral side has {ipsi.mean() / (contra.mean() + 1e-6):.1f}x more cells than contralateral")
        else:
            print(f"  ✗ UNEXPECTED: More cells on contralateral side")
    else:
        print(f"  ✗ NOT SIGNIFICANT: No ipsilateral restriction detected (p = {p_val:.4f})")
        print(f"    Cells may migrate bilaterally")
    # Save p-values 
    hypothesis_results.append({
        'hypothesis': f'H2_Ipsilateral_Restriction_{transplant}',
        't_stat': t_stat,
        'p_value': p_val,
        'cohens_d': cohens_d,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_ipsi': ipsi.mean(),
        'mean_contra': contra.mean(),
        'n_ipsi': len(ipsi),
        'n_contra': len(contra)
    })

# %%
######################
# Visualizations
#######################
norm_boxplots_with_stats = []

for transplant in data['transplant_type'].unique():
    subset = data[data['transplant_type'] == transplant].copy()
    
    # Calculate y position for brackets
    overall_max = subset['normalized_count'].max()
    bracket_y = overall_max * 1.10  # 10% above max
    text_y = overall_max * 1.15     # 15% above max
    
    # Calculate p-values for each tissue separately (ipsi vs contra within that tissue)
    p_value_by_tissue = {}
    for tissue in subset['tissue'].unique():
        tissue_data = subset[subset['tissue'] == tissue]
        ipsi = tissue_data[tissue_data['relative_side'] == 'ipsilateral']['normalized_count']
        contra = tissue_data[tissue_data['relative_side'] == 'contralateral']['normalized_count']
        
        if len(ipsi) > 0 and len(contra) > 0:
            # Welch's t-test (doesn't assume equal variances)
            t_stat, p_val = stats.ttest_ind(ipsi, contra, equal_var=False)
            p_value_by_tissue[tissue] = p_val
            
            # Print for verification
            print(f"{transplant} - {tissue}:")
            print(f"  Ipsilateral: n={len(ipsi)}, mean={ipsi.mean():.3f}")
            print(f"  Contralateral: n={len(contra)}, mean={contra.mean():.3f}")
            print(f"  t={t_stat:.3f}, p={p_val:.6f}")
            print()
    
    # Create annotation dataframes for brackets
    bracket_data = []
    text_data = []
    
    for tissue in subset['tissue'].unique():
        # Bracket line from ipsilateral to contralateral
        bracket_data.append({
            'tissue': tissue,
            'x': 'ipsilateral',
            'xend': 'contralateral', 
            'y': bracket_y,
            'yend': bracket_y
        })
        
        # Format p-value text
        p_value = p_value_by_tissue.get(tissue, None)
        if p_value is not None:
            if p_value < 0.0001:
                p_text = "p < 0.0001"
            elif p_value < 0.001:
                p_text = f"p = {p_value:.4f}"
            else:
                p_text = f"p = {p_value:.3f}"
        else:
            p_text = ""
            
        text_data.append({
            'tissue': tissue,
            'x': 1.5,  # Midpoint between categories (1=ipsi, 2=contra)
            'y': text_y,
            'label': p_text
        })
    
    bracket_df = pd.DataFrame(bracket_data)
    text_df = pd.DataFrame(text_data)
    
    # Ensure tissue is categorical with same levels as subset
    bracket_df['tissue'] = pd.Categorical(bracket_df['tissue'], 
                                           categories=subset['tissue'].unique())
    text_df['tissue'] = pd.Categorical(text_df['tissue'], 
                                        categories=subset['tissue'].unique())
    
    # Create the plot (NO free scales - shared y-axis)
    p = (pn.ggplot(subset, pn.aes(x='relative_side', y='normalized_count', fill='tissue'))
         + pn.geom_boxplot(alpha=0.8, outlier_shape='')  # Hide outliers since we show all points
         + pn.geom_jitter(
            pn.aes(shape='animal'),
            width=0.15,
            size=2,
            alpha=0.7,
            color='black',
        )
         + pn.facet_wrap('~tissue')  # Shared y-axis (no scales='free_y')
         + pn.theme_minimal()
         + pn.labs(
             title=f'Area Normalized F-ara-EdU\u207A Cell Counts for {transplant} Transplant',
             x='Relative Side',
             y='Normalized F-ara-EdU\u207A Cell Count\n(cells/ 100,000 µm²)'
         )
         + pn.scale_fill_brewer(type='qual', palette='Set1')
         + pn.scale_x_discrete(limits=['ipsilateral', 'contralateral'])  # Ensure order
         + pn.theme(
             legend_position='none',  # Fill is redundant with facet labels
             text=pn.element_text(family='DejaVu Sans'),
             figure_size=(7, 4),
             strip_text=pn.element_text(size=11, weight='bold')
         )
         # Add bracket lines
         + pn.geom_segment(
             pn.aes(x='x', xend='xend', y='y', yend='yend'),
             data=bracket_df,
             inherit_aes=False,
             size=0.5,
             color='black'
         )
         # Add p-value text
         + pn.geom_text(
             pn.aes(x='x', y='y', label='label'),
             data=text_df,
             inherit_aes=False,
             size=9,
             ha='center',
             va='bottom'
         )
    )
    
    norm_boxplots_with_stats.append(p)
    p.show()

#%% Alternate with additional significance brackets for tissue comparison

norm_boxplots_with_stats = []

for transplant in data['transplant_type'].unique():
    subset = data[data['transplant_type'] == transplant].copy()
    
    # Create a combined group variable for x-axis
    # Order: CB ipsi, CB contra, OL ipsi, OL contra
    subset['group'] = subset['tissue'] + '\n' + subset['relative_side'].astype(str)
    
    # Define the order
    group_order = [
        'Central Brain\nipsilateral',
        'Central Brain\ncontralateral', 
        'Optic Lobe\nipsilateral',
        'Optic Lobe\ncontralateral'
    ]
    subset['group'] = pd.Categorical(subset['group'], categories=group_order, ordered=True)
    
    # Calculate statistics
    overall_max = subset['normalized_count'].max()
    
    # --- Calculate p-values ---
    
    # 1. CB ipsi vs CB contra
    cb_ipsi = subset[(subset['tissue'] == 'Central Brain') & 
                     (subset['relative_side'] == 'ipsilateral')]['normalized_count']
    cb_contra = subset[(subset['tissue'] == 'Central Brain') & 
                       (subset['relative_side'] == 'contralateral')]['normalized_count']
    _, p_cb_side = stats.ttest_ind(cb_ipsi, cb_contra, equal_var=False)
    
    # 2. OL ipsi vs OL contra
    ol_ipsi = subset[(subset['tissue'] == 'Optic Lobe') & 
                     (subset['relative_side'] == 'ipsilateral')]['normalized_count']
    ol_contra = subset[(subset['tissue'] == 'Optic Lobe') & 
                       (subset['relative_side'] == 'contralateral')]['normalized_count']
    _, p_ol_side = stats.ttest_ind(ol_ipsi, ol_contra, equal_var=False)
    
    # 3. CB ipsi vs OL ipsi (the new comparison)
    _, p_tissue_ipsi = stats.ttest_ind(cb_ipsi, ol_ipsi, equal_var=False)
    
    # Print for verification
    print(f"\n{'='*60}")
    print(f"{transplant}")
    print(f"{'='*60}")
    print(f"CB ipsi vs CB contra: p = {p_cb_side:.6f}")
    print(f"OL ipsi vs OL contra: p = {p_ol_side:.6f}")
    print(f"CB ipsi vs OL ipsi:   p = {p_tissue_ipsi:.6f}")
    
    # --- Format p-values ---
    def format_pval(p):
        if p < 0.0001:
            return "p < 0.0001"
        elif p < 0.001:
            return f"p = {p:.4f}"
        else:
            return f"p = {p:.3f}"
    
    # --- Create bracket dataframes ---
    # Bracket heights (staggered to avoid overlap)
    y1 = overall_max * 1.05   # Within CB bracket
    y2 = overall_max * 1.05   # Within OL bracket  
    y3 = overall_max * 1.20   # Across tissues bracket (higher)
    
    text_offset = overall_max * 0.03
    
    # Using numeric x positions: 1=CB ipsi, 2=CB contra, 3=OL ipsi, 4=OL contra
    bracket_data = pd.DataFrame([
        # CB ipsi vs CB contra (positions 1 to 2)
        {'x': 1, 'xend': 2, 'y': y1, 'yend': y1},
        # OL ipsi vs OL contra (positions 3 to 4)
        {'x': 3, 'xend': 4, 'y': y2, 'yend': y2},
        # CB ipsi vs OL ipsi (positions 1 to 3)
        {'x': 1, 'xend': 3, 'y': y3, 'yend': y3},
    ])
    
    text_data = pd.DataFrame([
        {'x': 1.5, 'y': y1 + text_offset, 'label': format_pval(p_cb_side)},
        {'x': 3.5, 'y': y2 + text_offset, 'label': format_pval(p_ol_side)},
        {'x': 2.0, 'y': y3 + text_offset, 'label': format_pval(p_tissue_ipsi)},
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
             title=f'Area Normalized F-ara-EdU\u207A Cell Counts for {transplant} Transplant',
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

#%% Save normalized boxplots with stats
width = 7
height= 5
for i,transplant in enumerate(["Anterior-Anterior","Posterior-Posterior"]):
    norm_boxplots_with_stats[i] = norm_boxplots_with_stats[i] + pn.theme(figure_size=(width, height))
    norm_boxplots_with_stats[i].save(f"Figure_7_transplant_quant_boxplots_with_stats_{transplant}.pdf")


# %%
# Heatmap showing mean normalized counts for each combination

# Calculate means for each combination
heatmap_data = data.groupby(['transplant_type', 'tissue', 'relative_side'])['normalized_count'].mean().reset_index()
heatmap_data['label_text'] = heatmap_data['normalized_count'].round(1).astype(str)

p_heatmap = (
    pn.ggplot(heatmap_data, pn.aes(x='tissue', y='transplant_type', fill='normalized_count'))
    + pn.geom_tile(color='white', size=1.5)
    + pn.geom_text(pn.aes(label='label_text'), size=10, color='white', fontweight='bold')
    + pn.facet_wrap('~relative_side')
    + pn.scale_fill_gradient2(
        low='#2166ac', mid='#f7f7f7', high='#b2182b',
        midpoint=heatmap_data['normalized_count'].median(),
        name='Mean\nCells/100k µm²'
    )
    + pn.theme_minimal()
    + pn.labs(
        title='Transplant Cell Distribution Heatmap',
        subtitle='Mean normalized cell counts by transplant type, tissue, and side',
        x='Target Tissue',
        y='Transplant Type'
    )
    + pn.theme(
        axis_text_x=pn.element_text(angle=45, ha='right'),
        figure_size=(6, 4),
        legend_position='right'
    )
)
p_heatmap.show()

# %%
