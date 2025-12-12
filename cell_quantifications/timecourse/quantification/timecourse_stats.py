#! /usr/bin/env python3
# mamba activate edu_quant

#%% Imports
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices, bs
import plotnine as pn
from scipy import stats

#%% Load dataset
data = pd.read_csv('20251203_timecourse_quant.csv')

data

#%% Convert 'edu_cell_count' column to (edu_cell_count/area)*1000000 and make a float
scaling_factor = 1e6 #(in mm**2)
scaling_factor = 1e5  #(in 100000 um**2)
data['normalized_count'] = (data['edu_cell_count'] / data['area']) * scaling_factor

# create an 'animal' column by combining 'day' and 'replicate'
data['animal'] = data['day'].astype(str) + ' ' + data['replicate'].astype(str)

#%%
data = data.dropna()

#%% Create a numeric day column
day_mapping = {'0 day': 0, 
               '1 day': 1, 
               '2 day': 2, 
               '3 day': 3, 
               '4 day': 4, 
               '5 day': 5,
               '6 day': 6, 
               '7 day': 7}

data['day_numeric'] = data['day'].map(day_mapping)

# %% Create a faceted barplot of raw cell counts
p = (
    pn.ggplot(
        data,
        pn.aes(x='day', y='edu_cell_count', fill='region')
        ) 
    + pn.geom_boxplot(position='dodge')
    # overlay jittered points
    + pn.geom_smooth(pn.aes(color='tissue'),method = "loess", group='tissue', se = True ,size=0.75, linetype='dashed')
    + pn.geom_jitter(position=pn.position_jitterdodge(jitter_width=0.1, dodge_width=0.75), size=0.5, alpha=1)
    + pn.facet_wrap('~tissue', scales='free_x')
    + pn.theme_minimal()
    + pn.scale_fill_brewer(type='qual', palette='Set1')
    + pn.scale_color_brewer(type='qual', palette='Set1')
    + pn.labs(y='F-ara-EdU\u207A Cells\n(raw counts)')
    + pn.ggtitle('Raw F-ara-EdU\u207A Cell Counts Over Time')
    + pn.theme(legend_position='top', 
        legend_box='horizontal',
        axis_text_x=pn.element_text(rotation=-90, hjust=1)
        )
)

p.show()

# %% Create a faceted barplot of area-normalized cell counts
norm_boxplot = (
    pn.ggplot(
        data,
        pn.aes(x='day', y='normalized_count', fill='region')
        ) 
    + pn.geom_boxplot()
    + pn.geom_smooth(pn.aes(color='tissue'),method = "loess", group='tissue', se = True ,size=0.75, linetype='dashed')
    + pn.geom_jitter(position=pn.position_jitterdodge(jitter_width=0.1, dodge_width=0.75), size=0.5, alpha=1)
    #+ pn.facet_grid('orientation~tissue', scales='free')
    + pn.facet_wrap('~tissue', scales='free_x')
    + pn.theme_minimal()
    + pn.scale_fill_brewer(type='qual', palette='Set1')
    + pn.scale_color_brewer(type='qual', palette='Set1')
    # change y axis label
    + pn.labs(y='Normalized F-ara-EdU\u207A Cells\n(cells / 100,000 µm²)')
    # remove vertical lines
    + pn.theme(panel_grid_major_x=pn.element_blank())
    + pn.ggtitle('Area-Normalized F-ara-EdU\u207A Cell Counts Over Time')
    + pn.theme(legend_position='top', 
        legend_box='horizontal',
        axis_text_x=pn.element_text(rotation=-90, hjust=1)
        )
)

norm_boxplot.show()

#%%
#################
# OLS Model
#################

# %% Fit a linear model using statsmodels to the normalized_count data with day and tissue as predictors
y, X = dmatrices('normalized_count ~ C(day) * C(tissue)', data=data, return_type='dataframe')
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

#%% Residuals vs fitted 
fitted_vals = results.fittedvalues
residuals = results.resid

residuals_df = pd.DataFrame({'fitted': fitted_vals, 'residuals': residuals})
p = (
    pn.ggplot(residuals_df, pn.aes(x='fitted', y='residuals'))
    + pn.geom_point()
    + pn.geom_hline(yintercept=0, color='black')
    + pn.geom_smooth(method='loess', color='red', linetype='dashed', size=0.75)
    + pn.theme_minimal()
    + pn.labs(x='Fitted Values', y='Residuals')
    + pn.ggtitle('Residuals vs Fitted Values')
)

p.show()

#%% Influence plot
#sm.graphics.influence_plot(results, criterion="cooks")

#%% Compare the true relationship to OLS predictions with confidence intervals.
pred_ols = results.get_prediction()
alpha = 0.05
pred_summary = pred_ols.summary_frame(alpha=alpha)

#%%
data['predicted_ols'] = pred_summary['mean']
data['ci_lower_ols'] = pred_summary['obs_ci_lower']
data['ci_upper_ols'] = pred_summary['obs_ci_upper']

#%%
ols_fit_scatter = (
    pn.ggplot(data, pn.aes(x='day', y='normalized_count',color='tissue',fill='tissue'))
    + pn.geom_jitter(size=1, alpha=0.5, position=pn.position_jitter(width=0.2, height=0))
    + pn.geom_line(pn.aes(y='predicted_ols', group='tissue'), size=1, linetype='dashed')
    + pn.geom_ribbon(pn.aes(ymin='ci_lower_ols', ymax='ci_upper_ols', group='tissue'), alpha=0.1)
    + pn.theme_minimal()
    + pn.labs(y='Normalized F-ara-EdU\u207A Cells\n(cells / 100,000 µm²)')
    + pn.scale_color_brewer(type='qual', palette='Set1')
    + pn.scale_fill_brewer(type='qual', palette='Set1')
    + pn.facet_wrap('~tissue', scales='free_x')
    + pn.ggtitle(f'OLS Model Predictions with {int((1-alpha)*100)}% Confidence Intervals')
    + pn.theme(legend_position='top', legend_box='horizontal')
)
ols_fit_scatter.show()

#%%
################
# Mixed Effects Model
################
# %% Fit a mixed effects model as above but add a random effect for animal
data['log_normalized_count'] = np.log(data['normalized_count'] + 1)
data['day_numeric_sq'] = data['day_numeric'] ** 2
#model_formula = 'log_normalized_count ~ C(day) * C(tissue)'
model_formula = 'normalized_count ~ C(day) * C(tissue)'

#model_formula = 'normalized_count ~ C(day) * C(tissue) + day_numeric' # Does this approximate ARIMA to account for time autoregression?
md = sm.MixedLM.from_formula(model_formula,
                             groups='animal', 
                             #re_formula='~day_numeric',
                             data=data)  # Fit separately for each tissue
mdf = md.fit()
print(mdf.summary())

#%% Get coefficients with confidence intervals
conf_int = mdf.conf_int()
params = mdf.params

results_df = pd.DataFrame({
    'Coefficient': params,
    'CI_lower': conf_int[0],
    'CI_upper': conf_int[1],
    'p-value': mdf.pvalues
})
print(results_df)

# %% Compare the true relationship to MixedLM predictions with confidence intervals.
data['predicted_mixed'] = mdf.predict(data)

# Use the model's own design matrix
X = mdf.model.exog
X_df = pd.DataFrame(X, columns=mdf.model.exog_names)

# Get covariance matrix - only for FIXED effects
cov_params = mdf.cov_params()

# Filter out random effects parameters (they contain 'Var' or 'Cov')
fixed_effects_mask = ~cov_params.index.str.contains('Var|Cov', regex=True)
cov_params_fixed = cov_params.loc[fixed_effects_mask, fixed_effects_mask]

print(f"X columns: {X_df.columns.tolist()}")
print(f"Fixed effects params: {cov_params_fixed.index.tolist()}")
print(f"Cov params shape: {cov_params_fixed.shape}")

# Align the columns
X_aligned = X_df[cov_params_fixed.index]

# Convert to numpy
X_array = X_aligned.values
cov_params_array = cov_params_fixed.values

# %% Calculate standard errors for predictions
pred_var = np.diag(X_array @ cov_params_array @ X_array.T)
pred_se = np.sqrt(pred_var)

# %% Calculate confidence intervals of the mean
z_crit = stats.norm.ppf(1 - alpha/2)

data['ci_lower_mixed'] = data['predicted_mixed'] - z_crit * pred_se
data['ci_upper_mixed'] = data['predicted_mixed'] + z_crit * pred_se

# %% Calculate prediction intervals for new observations

# Add residual variance for prediction intervals
residual_var = mdf.scale  # residual variance from the model
pred_se_pred = np.sqrt(pred_var + residual_var)

# Prediction intervals for NEW OBSERVATIONS
data['pi_lower_mixed'] = data['predicted_mixed'] - z_crit * pred_se_pred
data['pi_upper_mixed'] = data['predicted_mixed'] + z_crit * pred_se_pred

#%%
# Extract p-values from the model for day effects
# The model coefficients are already contrasts vs Day 0 (the reference level)

# Get parameters and p-values for day effects
day_params = mdf.params.filter(like='C(day)')
day_pvals = mdf.pvalues.filter(like='C(day)')

# Also get interaction terms if you want tissue-specific comparisons
interaction_params = mdf.params.filter(like='C(day)[T.')
interaction_pvals = mdf.pvalues.filter(like='C(day)[T.')

print("\nDay effects (main effects - applies to reference tissue):")
print(pd.DataFrame({'Coefficient': day_params, 'p-value': day_pvals}))

print("\nInteraction effects:")
print(pd.DataFrame({'Coefficient': interaction_params, 'p-value': interaction_pvals}))

# Create significance labels for each day and tissue
sig_annotations = []

# Days to compare (all except reference '0 day')
days_to_test = ['1 day', '2 day', '3 day', '4 day', '5 day', '6 day','7 day']

for tissue in data['tissue'].unique():
    for day in days_to_test:
        # Determine which p-value to use
        # For reference tissue (Central Brain), use main effect
        # For Optic Lobe, combine main effect + interaction
        
        main_effect_name = f'C(day)[T.{day}]'
        interaction_name = f'C(day)[T.{day}]:C(tissue)[T.Optic Lobe]'
        
        if tissue == 'Central Brain':
            # Use main effect p-value
            if main_effect_name in day_pvals.index:
                pval = day_pvals[main_effect_name]
            else:
                continue
        else:  # Optic Lobe
            # For interaction model, you need to test the combined effect
            # This is more complex - for now, use the interaction p-value as approximation
            if interaction_name in interaction_pvals.index:
                pval = interaction_pvals[interaction_name]
            elif main_effect_name in day_pvals.index:
                # If no interaction term, same as main effect
                pval = day_pvals[main_effect_name]
            else:
                continue
        
        # Create significance label
        if pval < 0.001:
            sig = '***'
        elif pval < 0.01:
            sig = '**'
        elif pval < 0.05:
            sig = '*'
        else:
            sig = 'NS'
        
        # Get y position for annotation
        subset = data[(data['tissue'] == tissue) & (data['day'] == day)]
        if len(subset) > 0:
            y_pos = subset['normalized_count'].max() * 1.1
            
            sig_annotations.append({
                'tissue': tissue,
                'day': day,
                'y_pos': y_pos,
                'label': sig,
                'p_value': pval
            })

sig_df = pd.DataFrame(sig_annotations)
print("\nSignificance annotations:")
print(sig_df)

#%%
mm_fit_scatter = (
    pn.ggplot(data, pn.aes(x='day', y='normalized_count', color='tissue', fill='tissue'))
    + pn.geom_jitter(size=1, alpha=0.5, position=pn.position_jitter(width=0.2, height=0))
    + pn.geom_line(pn.aes(y='predicted_mixed', group='tissue'), size=0.75, linetype='dashed')
    + pn.geom_ribbon(pn.aes(ymin='ci_lower_mixed', ymax='ci_upper_mixed', group='tissue'), alpha=0.1)
    # Add significance markers
    + pn.geom_text(sig_df, pn.aes(x='day', y='y_pos', label='label'), 
                   size=8, color='black', va='bottom')
    + pn.theme_minimal()
    +  pn.labs(
        title='Mixed Effects Model: Tissue-Specific Significance vs Day 0',
        subtitle='* p<0.05, ** p<0.01, *** p<0.001',
        y='Normalized F-ara-EdU\u207A Cells\n(cells / 100,000 µm²)'
    )
    + pn.scale_color_brewer(type='qual', palette='Set1')
    + pn.scale_fill_brewer(type='qual', palette='Set1')
    + pn.facet_wrap('~tissue', scales='free')
    + pn.theme(legend_position='top', 
               legend_box='horizontal',
               axis_text_x=pn.element_text(rotation=-90, hjust=1))
)
mm_fit_scatter.show()

#%% Residuals vs fitted for MixedLM
residuals_mixed = data['normalized_count'] - data['predicted_mixed']
data['residuals_mixed'] = residuals_mixed

# Q-Q plot for normality
mm_qq = (
    pn.ggplot(data, pn.aes(sample='residuals_mixed'))
    + pn.stat_qq()
    + pn.stat_qq_line(color='red')
    + pn.theme_minimal()
    + pn.ggtitle('Q-Q Plot of Mixed Effects Model Residuals')
    
)
mm_qq.show()

# Residuals vs fitted
mm_resid = (
    pn.ggplot(data, pn.aes(x='predicted_mixed', y='residuals_mixed'))
    + pn.geom_point()
    + pn.geom_hline(yintercept=0, color='red', linetype='dashed')
    + pn.geom_smooth(method='loess', se=False)
    + pn.theme_minimal()
    #+ pn.facet_wrap('~tissue')
    + pn.ggtitle('Residuals vs Fitted - Mixed Effects Model')
)
mm_resid.show()

#%%
#########################
if 'day_numeric' not in sig_df.columns:
    sig_df['day_numeric'] = sig_df['day'].map(day_mapping)

# Create brackets with staggered heights to prevent overlap
bracket_data = []
text_data = []

# Group by tissue to handle each facet separately
for tissue in sig_df['tissue'].unique():
    tissue_sig = sig_df[sig_df['tissue'] == tissue].copy()
    tissue_data = data[data['tissue'] == tissue]
    
    # Get base height for brackets
    y_max = tissue_data['normalized_count'].max()
    
    # Sort by day to determine stacking order
    tissue_sig = tissue_sig.sort_values('day_numeric')
    
    for idx, (_, row) in enumerate(tissue_sig.iterrows()):
        day_num = row['day_numeric']
        label = row['label']
        
        # Skip day 0
        if day_num == 0:
            continue
        
        # Calculate bracket height with better spacing
        # Stagger based on day_num to prevent overlap
        # Use larger increments for better separation
        base_height = y_max * 0.7
        height_increment = y_max * 0.08  # Increased from 0.02 to 0.08
        y_height = base_height + (day_num - 1) * height_increment
        
        # Numeric positions
        x_start = 0
        x_end = day_num
        x_mid = (x_start + x_end) / 2
        
        # Tick size
        tick_size = y_max * 0.02
        
        # Horizontal line
        bracket_data.append({
            'tissue': tissue,
            'x': x_start,
            'xend': x_end,
            'y': y_height,
            'yend': y_height
        })
        
        # Left tick
        bracket_data.append({
            'tissue': tissue,
            'x': x_start,
            'xend': x_start,
            'y': y_height - tick_size,
            'yend': y_height
        })
        
        # Right tick
        bracket_data.append({
            'tissue': tissue,
            'x': x_end,
            'xend': x_end,
            'y': y_height - tick_size,
            'yend': y_height
        })
        
        # Text (slightly above bracket)
        text_data.append({
            'tissue': tissue,
            'x': x_mid,
            'y': y_height + tick_size,
            'label': label
        })

bracket_df = pd.DataFrame(bracket_data)
text_df = pd.DataFrame(text_data)

# Plot with improved spacing
mm_fit_scatter_brackets = (
    pn.ggplot(data, pn.aes(x='day_numeric', y='normalized_count', color='tissue', fill='tissue'))
    + pn.geom_jitter(size=1, alpha=0.5, position=pn.position_jitter(width=0.15, height=0))
    + pn.geom_line(pn.aes(y='predicted_mixed', group='tissue'), size=0.75, linetype='dashed')
    + pn.geom_ribbon(pn.aes(ymin='ci_lower_mixed', ymax='ci_upper_mixed', group='tissue'), alpha=0.1)
    # Add significance brackets
    + pn.geom_segment(bracket_df, 
                     pn.aes(x='x', xend='xend', y='y', yend='yend'),
                     color='black', size=0.5, inherit_aes=False)
    # Add text
    + pn.geom_text(text_df,
                   pn.aes(x='x', y='y', label='label'),
                   size=9, color='black', fontweight='bold', inherit_aes=False)
    + pn.theme_minimal()
    + pn.labs(
        title='Mixed Effects Model: Tissue-Specific Significance vs Day 0',
        subtitle='* p<0.05, ** p<0.01, *** p<0.001',
        x='Day',
        y='Normalized F-ara-EdU\u207A Cells\n(cells / 100,000 µm²)'
    )
    + pn.scale_x_continuous(
        breaks=[0, 1, 2, 3, 4, 5, 6, 7],
        labels=['0 day', '1 day', '2 day', '3 day', '4 day', '5 day', '6 day', '7 day']
    )
    + pn.scale_color_brewer(type='qual', palette='Set1')
    + pn.scale_fill_brewer(type='qual', palette='Set1')
    + pn.facet_wrap('~tissue',scales='free_y')
    + pn.theme(
        legend_position='top',
        legend_box='horizontal',
        axis_text_x=pn.element_text(angle=45, ha='right')
    )
)
mm_fit_scatter_brackets.show()

#########################

#%%
# #%% Create a prediction grid with all day levels and both tissues
# unique_days = data['day'].unique()
# pred_grid = pd.DataFrame({
#     'day': np.repeat(unique_days, 2),
#     'tissue': ['Central Brain', 'Optic Lobe'] * len(unique_days),
#     'animal': '0 day embryo 1'  # Use a reference animal
# })

# # Get predictions
# pred_grid['predicted'] = mdf.predict(pred_grid)

# # For categorical x-axis, need to convert to numeric for plotting lines
# # Use the day_mapping you created earlier
# pred_grid['day_numeric'] = pred_grid['day'].map(day_mapping)
# pred_grid = pred_grid.sort_values(['tissue', 'day_numeric'])

# # Plot with both data and predictions
# p = (
#     pn.ggplot()
#     + pn.geom_point(data, pn.aes(x='day', y='normalized_count', color='tissue'), alpha=0.5, size=2)
#     + pn.geom_line(pred_grid, pn.aes(x='day', y='predicted', color='tissue', group='tissue'), size=1.5)
#     + pn.geom_point(pred_grid, pn.aes(x='day', y='predicted', color='tissue'), size=3, shape='D')
#     + pn.theme_minimal()
#     + pn.labs(x='Day', y='Normalized F-ara-EdU\u207A Cells\n(cells / 100,000 µm²)')
#     + pn.facet_wrap('~tissue',)
#     + pn.scale_color_brewer(type='qual', palette='Set1')
#     + pn.ggtitle('Mixed Model Predictions (Categorical Day)')
#     + pn.theme(legend_position='top')
# )
# p.show()

#%%
################
# Quadratic Model
################
# Add quadratic term
# data['day_numeric_sq'] = data['day_numeric'] ** 2

# formula = 'normalized_count ~ day_numeric + day_numeric_sq + C(tissue)'
# md = sm.MixedLM.from_formula(formula, groups='animal', 
#                               re_formula='~day_numeric', data=data)
# mdf = md.fit()
# print(mdf.summary())

# # The inflection point (where acceleration starts) for a quadratic is at the vertex
# # For y = a + b*x + c*x^2, vertex is at x = -b/(2*c)
# b = mdf.params['day_numeric']
# c = mdf.params['day_numeric_sq']
# inflection_day = -b / (2 * c)
# print(f"\nInflection day (quadratic): {inflection_day:.2f}")

# # Visualize
# pred_days = np.linspace(0, 7, 100)
# pred_grid = pd.DataFrame({
#     'day_numeric': np.tile(pred_days, 2),
#     'day_numeric_sq': np.tile(pred_days**2, 2),
#     'tissue': np.repeat(['Central Brain', 'Optic Lobe'], len(pred_days)),
#     'animal': '0 day embryo 1'
# })
# pred_grid['predicted'] = mdf.predict(pred_grid)

# p = (
#     pn.ggplot()
#     + pn.geom_point(data, pn.aes(x='day_numeric', y='normalized_count', color='tissue'), alpha=0.5)
#     + pn.geom_line(pred_grid, pn.aes(x='day_numeric', y='predicted', color='tissue'), size=1.5)
#     + pn.geom_vline(xintercept=inflection_day, linetype='dashed', color='red')
#     + pn.theme_minimal()
#     + pn.scale_color_brewer(type='qual', palette='Set1')
#     + pn.labs(x='Day', y='Normalized Count', title=f'Quadratic Model (Inflection ≈ day {inflection_day:.1f})')
# )
# p.show()

#%%
#############
# Spline model
#############

#%% Central Brain only
# Create spline basis (3 degrees of freedom allows for inflection)
# This creates smooth curves that can change direction
formula = 'normalized_count ~ bs(day_numeric, df=3)'
md = sm.MixedLM.from_formula(formula, 
                             groups='animal',
                             #re_formula='~day_numeric',
                             data=data[data['tissue'] == 'Central Brain']
                             )
mdf = md.fit()

# To find inflection, calculate second derivative numerically
pred_days = np.linspace(0, 7, 100)
pred_grid_cb = pd.DataFrame({
    'day_numeric': pred_days,
    'animal': '0 day embryo 1',
    'tissue': 'Central Brain'
})
pred_grid_cb['predicted'] = mdf.predict(pred_grid_cb)

# Calculate second derivative (where curvature changes from negative to positive)
second_deriv = np.gradient(np.gradient(pred_grid_cb['predicted']))
zero_crossings = np.where(np.diff(np.sign(second_deriv)))[0]

if len(zero_crossings) > 0:
    inflection_idx_cb = zero_crossings[0]
    inflection_day_cb = pred_days[inflection_idx_cb]
    print(f"Inflection day (spline, Central Brain): {inflection_day_cb:.2f}")

#%% Optic Lobe only
md = sm.MixedLM.from_formula(formula, 
                             groups='animal', 
                             #re_formula='~day_numeric',
                             data=data[data['tissue'] == 'Optic Lobe']
                             )
mdf = md.fit()
pred_grid_ol = pd.DataFrame({
    'day_numeric': pred_days,
    'animal': '0 day embryo 1',
    'tissue': 'Optic Lobe'
})
pred_grid_ol['predicted'] = mdf.predict(pred_grid_ol)

# Calculate second derivative (where curvature changes from negative to positive)
second_deriv_ol = np.gradient(np.gradient(pred_grid_ol['predicted']))
zero_crossings_ol = np.where(np.diff(np.sign(second_deriv_ol)))[0]

if len(zero_crossings_ol) > 0:
    inflection_idx_ol = zero_crossings_ol[0]
    inflection_day_ol = pred_days[inflection_idx_ol]
    print(f"Inflection day (spline, Optic Lobe): {inflection_day_ol:.2f}")

#%% Visualize spline model
pred_grid_combined = pd.concat([pred_grid_cb, pred_grid_ol], ignore_index=True)
spline_inflection_fit = (
    pn.ggplot()
    + pn.geom_jitter(data, pn.aes(x='day_numeric', y='normalized_count', color='tissue'), position=pn.position_jitterdodge(jitter_width=0.2, dodge_width=0.3), size=0.5,alpha=0.5)
    + pn.geom_line(pred_grid_combined, pn.aes(x='day_numeric', y='predicted', color='tissue'), size=1.5)
    + pn.geom_vline(xintercept=inflection_day_cb, linetype='dashed', color='red')
    + pn.geom_vline(xintercept=inflection_day_ol, linetype='dashed', color='blue')
    # Add text annotations along vline for each calculated inflection point
    + pn.annotate('text', x=inflection_day_cb, y=data['normalized_count'].max() * 0.9, 
                  label=f'CB Inflection: day {inflection_day_cb:.2f}', 
                  color='red', size=8, angle=90, ha='right',va='top')
    + pn.annotate('text', x=inflection_day_ol, y=data['normalized_count'].max() * 0.8, 
                  label=f'OL Inflection: day {inflection_day_ol:.2f}', 
                  color='blue', size=8, angle=90, ha='right',va='top')
    + pn.theme_minimal()
    + pn.scale_color_brewer(type='qual', palette='Set1')
    #+ pn.facet_wrap('~tissue')
    + pn.labs(x='Day', y='Normalized Count', title=f'Spline Model for Inflection')
    + pn.theme(legend_position='top', legend_box='horizontal')
)
spline_inflection_fit.show()

#%%
#####################
# Create quantification supplemental figure
#####################
supp_figure =  mm_fit_scatter_brackets / mm_qq / mm_resid  / spline_inflection_fit
width = 7
height= 18
supp_figure = supp_figure + pn.theme(figure_size=(width, height))

supp_figure.show()

#%%
supp_figure.save('Figure_S2_timecourse_quantification.pdf', width=width, height=height)

#%%
norm_boxplot.save('Figure_1E_timecourse_quant_boxplot.pdf', width=6, height=4)
################
# Output
################
# %% Save the data with predictions and CIs to a new CSV
data.to_csv('202511203_timecourse_quant_with_model_predictions.csv', index=False)

#%%