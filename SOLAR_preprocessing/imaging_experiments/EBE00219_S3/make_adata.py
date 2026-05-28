from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import polars as pl
import seaborn as sns

#from fishtools.postprocess import (
#    jitter,
#    normalize_pearson,
#    normalize_total,
#    rotate_label,
#)
#from fishtools.utils.io import Workspace
#from fishtools.utils.plot import add_scale_bar, plot_embedding, plot_wheel, plot_with_hist
#from fishtools.utils.utils import copy_signature, create_rotation_matrix

# %%

mpl.rcParams["figure.dpi"] = 300

sns.set_theme()
path = Path("/warm/raw/20250612_ebe00219_3/analysis/deconv")
rois = ["big","small"]  
seg_codebook = "edu"
codebooks = ["ebe_tricycle_targets", "ebe_devprobeset_targets"] 

output_dir = path / "analysis_output"
output_dir.mkdir(exist_ok=True)
#raw_decoded = pl.read_parquet(ws.deconved / f"left+{codebooks[0]}.parquet")



# %%
dfs = {}
for roi in rois:
    glob_path = path / f"stitch--{roi}+{seg_codebook}" / "chunks+*/ident_*.parquet"
    df_roi = (
        pl.scan_parquet(
            glob_path,
            include_file_paths="path",
            #allow_missing_columns=True,
            missing_columns="insert",
        )
        .with_columns(
            z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
            codebook=pl.col("path").str.extract(r"chunks\+(\w+)"),
            spot_id=pl.col("spot_id").cast(pl.UInt32),
            roi=pl.lit(roi),
        )
        .with_columns(
            roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
        )
        .sort("z")
        .collect()
    )
    if not df_roi.is_empty():
        dfs[roi] = df_roi

if not dfs:
    raise ValueError(
        f"No ident files found for any ROI in '{rois}' with seg_codebook '{seg_codebook}', or 'rois' list is empty."
    )
df = pl.concat(dfs.values())

# %%
intensity_types = ["edu", "reddot"] 

all_intensities = {}
for roi in rois:
    print(f"\nProcessing ROI: {roi}")
    roi_intensities = {}
    for intensity in intensity_types:
        glob_path_intensity = (
            path / f"stitch--{roi}+{seg_codebook}" / f"intensity_{intensity}/intensity-*.parquet"
        )
        print(f"  Looking for {intensity} intensity files at: {glob_path_intensity}")
        
        # Check if the directory exists
        intensity_dir = glob_path_intensity.parent
        if not intensity_dir.exists():
            print(f"    Directory does not exist: {intensity_dir}")
            continue
            
        # Check if any matching files exist
        matching_files = list(intensity_dir.glob("intensity-*.parquet"))
        print(f"    Found {len(matching_files)} intensity files")
        
        try:
            intensity_roi = (
                pl.scan_parquet(
                    glob_path_intensity,
                    include_file_paths="path",
                    allow_missing_columns=True,
                )
                .with_columns(
                    z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
                    roi=pl.lit(roi),
                )
                .with_columns(
                    roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
                    roilabelz=pl.col("z").cast(pl.Utf8) + pl.col("roi") + pl.col("label").cast(pl.Utf8),
                )
                .collect()
            )
            if not intensity_roi.is_empty():
                print(f"    Successfully loaded {len(intensity_roi)} rows for {intensity}")
                roi_intensities[intensity] = intensity_roi
            else:
                print(f"    Loaded dataframe is empty for {intensity}")
        except Exception as e:
            print(f"    Error loading {intensity} intensity files for ROI '{roi}': {e}")
            pass
    
    if roi_intensities:  # Only add if we found some intensities for this ROI
        all_intensities[roi] = roi_intensities
        print(f"  Added {len(roi_intensities)} intensity types for ROI {roi}")
    else:
        print(f"  No intensity data found for ROI {roi}")

print(f"\nSummary: Found intensity data for {len(all_intensities)} ROIs")
for roi, intensities in all_intensities.items():
    print(f"  {roi}: {list(intensities.keys())}")

if not all_intensities:
    raise ValueError(
        f"No intensity files found for any ROI in '{rois}' with any intensity type in {intensity_types}."
    )

# %%
intensity_types = ["edu", "pi"]  # Define multiple intensity types

all_intensities = {}
for roi in rois:
    print(f"\nProcessing ROI: {roi}")
    roi_intensities = {}
    for intensity in intensity_types:
        glob_path_intensity = (
            path / f"stitch--{roi}+{seg_codebook}" / f"intensity_{intensity}/intensity-*.parquet"
        )
        print(f"  Looking for {intensity} intensity files at: {glob_path_intensity}")
        
        # Check if the directory exists
        intensity_dir = glob_path_intensity.parent
        if not intensity_dir.exists():
            print(f"    Directory does not exist: {intensity_dir}")
            continue
            
        # Check if any matching files exist
        matching_files = list(intensity_dir.glob("intensity-*.parquet"))
        print(f"    Found {len(matching_files)} intensity files")
        
        try:
            intensity_roi = (
                pl.scan_parquet(
                    glob_path_intensity,
                    include_file_paths="path",
                    allow_missing_columns=True,
                )
                .with_columns(
                    z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8),
                    roi=pl.lit(roi),
                )
                .with_columns(
                    roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
                    roilabelz=pl.col("z").cast(pl.Utf8) + pl.col("roi") + pl.col("label").cast(pl.Utf8),
                )
                .collect()
            )
            if not intensity_roi.is_empty():
                print(f"    Successfully loaded {len(intensity_roi)} rows for {intensity}")
                roi_intensities[intensity] = intensity_roi
            else:
                print(f"    Loaded dataframe is empty for {intensity}")
        except Exception as e:
            print(f"    Error loading {intensity} intensity files for ROI '{roi}': {e}")
            pass
    
    if roi_intensities:  # Only add if we found some intensities for this ROI
        all_intensities[roi] = roi_intensities
        print(f"  Added {len(roi_intensities)} intensity types for ROI {roi}")
    else:
        print(f"  No intensity data found for ROI {roi}")

print(f"\nSummary: Found intensity data for {len(all_intensities)} ROIs")
for roi, intensities in all_intensities.items():
    print(f"  {roi}: {list(intensities.keys())}")

if not all_intensities:
    raise ValueError(
        f"No intensity files found for any ROI in '{rois}' with any intensity type in {intensity_types}."
    )

# %%

_spots_accumulator = defaultdict(list)

# Iterate over each ROI and its corresponding DataFrame from 'dfs'
# df_for_current_roi is dfs[roi]
joineds = []
for roi, _df in dfs.items():
    print(roi)
    df_cbs = []
    for codebook in _df["codebook"].unique():
        spots_file_path = path / f"{roi}+{codebook}.parquet"

        if not spots_file_path.exists():
            print(f"Info: Spots file not found, skipping: {spots_file_path}")
            continue

        current_roi_codebook_spots_df = pl.read_parquet(spots_file_path).with_columns(
            roi=pl.lit(roi), codebook=pl.lit(codebook)
        )

        if "index" not in current_roi_codebook_spots_df.columns:
            current_roi_codebook_spots_df = current_roi_codebook_spots_df.with_row_index(name="label")
        else:
            current_roi_codebook_spots_df = current_roi_codebook_spots_df.rename(dict(index="label"))
        df_cbs.append(current_roi_codebook_spots_df)

    df_for_current_roi = pl.concat(df_cbs)

    joined = (
        df.filter(pl.col("roi") == roi)
        .with_columns(label=pl.col("label").cast(pl.UInt32))
        .join(df_for_current_roi, on=[pl.col("codebook"), pl.col("label")], how="left")
    )

    if joined.is_empty():
        print(f"Info: No relevant ident data found in dfs['{roi}']")
    joineds.append(joined)
joined = pl.concat(joineds)
# %%
u = current_roi_codebook_spots_df.filter(~pl.col("label").is_in(joined["spot_id"]))

# %%
len(joined) / len(current_roi_codebook_spots_df)

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

#notincell = raw_decoded.filter(~pl.col("index").is_in(df["spot_id"]))
#plt.hist(notincell["z"], bins=100)
# %%
#plt.hist(raw_decoded.filter(pl.col("index").is_in(df["spot_id"]))["z"], bins=100)


# %%
def arrange_rois(polygons: pl.DataFrame, max_columns: int = 2, padding: float = 100):
    # Calculate bounding box for each ROI
    roi_bounds = (
        polygons.group_by("roi")
        .agg(
            min_x=pl.col("centroid_x").min(),
            max_x=pl.col("centroid_x").max(),
            min_y=pl.col("centroid_y").min(),
            max_y=pl.col("centroid_y").max(),
        )
        .sort("roi")
    )
    print(roi_bounds)

    # Calculate offsets for each ROI in a grid layout
    roi_offsets = {}
    max_width = 0
    max_height = 0

    for i, roi_row in enumerate(roi_bounds.iter_rows(named=True)):
        row = i // max_columns
        col = i % max_columns
        width = roi_row["max_x"] - roi_row["min_x"]
        height = roi_row["max_y"] - roi_row["min_y"]

        x_offset = col * (max_width + padding) - roi_row["min_x"]
        y_offset = row * (max_height + padding) - roi_row["min_y"]

        roi_offsets[roi_row["roi"]] = (x_offset, y_offset)
        max_width = max(max_width, width)
        max_height = max(max_height, height)

    # Apply offsets to polygons
    return polygons.with_columns(
        [
            pl.col("centroid_x")
            + pl.col("roi").map_elements(lambda r: roi_offsets[r][0], return_dtype=pl.Float64),
            pl.col("centroid_y")
            + pl.col("roi").map_elements(lambda r: roi_offsets[r][1], return_dtype=pl.Float64),
        ]
    ), roi_offsets


# Modified polygon processing to handle multiple intensities
polygons = {}
for roi in rois:
    polygons[roi] = (
        pl.scan_parquet(
            path / f"stitch--{roi}+{seg_codebook}" / f"chunks+{codebooks[0]}/polygons_*.parquet",
            include_file_paths="path",
        )
        .with_columns(z=pl.col("path").str.extract(r"(\d+)\.parquet").cast(pl.UInt8), roi=pl.lit(roi))
        .with_columns(
            roilabel=pl.col("roi") + pl.col("label").cast(pl.Utf8),
            roilabelz=pl.col("z").cast(pl.Utf8) + pl.col("roi") + pl.col("label").cast(pl.Utf8),
        )
        .drop("path")
        .sort("z")
        .collect()
    )
polygons = pl.concat(polygons.values())

# Join multiple intensities to polygons
print(f"\nJoining intensity data to polygons...")
print(f"Polygons shape before joining: {polygons.shape}")

# Initialize intensity columns with null values for all intensity types
for intensity_type in intensity_types:
    polygons = polygons.with_columns([
        pl.lit(None, dtype=pl.Float64).alias(f"mean_intensity_{intensity_type}"),
        pl.lit(None, dtype=pl.Float64).alias(f"max_intensity_{intensity_type}"),
        pl.lit(None, dtype=pl.Float64).alias(f"min_intensity_{intensity_type}")
    ])

# Now update with actual intensity data for each ROI
for roi in rois:
    print(f"\nProcessing ROI: {roi}")
    if roi in all_intensities:
        for intensity_type, intensity_data in all_intensities[roi].items():
            print(f"  Joining {intensity_type} intensity data...")
            print(f"  Intensity data shape: {intensity_data.shape}")
            
            # Check for overlapping roilabelz values
            roi_polygons = polygons.filter(pl.col("roi") == roi)
            polygons_roilabelz = set(roi_polygons["roilabelz"])
            intensity_roilabelz = set(intensity_data["roilabelz"])
            overlap = polygons_roilabelz.intersection(intensity_roilabelz)
            print(f"  Overlapping roilabelz values: {len(overlap)} out of {len(polygons_roilabelz)} polygon cells")
            
            # Select only the intensity columns we need
            intensity_cols = ["roilabelz", "mean_intensity", "max_intensity", "min_intensity"]
            available_cols = [col for col in intensity_cols if col in intensity_data.columns]
            print(f"  Available intensity columns: {available_cols}")
            
            intensity_data_subset = intensity_data.select(available_cols)
            
            # Create a mapping dataframe for this ROI and intensity type
            intensity_mapping = intensity_data_subset.rename({
                "mean_intensity": f"mean_intensity_{intensity_type}_temp",
                "max_intensity": f"max_intensity_{intensity_type}_temp", 
                "min_intensity": f"min_intensity_{intensity_type}_temp"
            })
            
            # Join and update the specific columns for this ROI
            polygons = (
                polygons
                .join(intensity_mapping, on="roilabelz", how="left")
                .with_columns([
                    # Update intensity columns only where roi matches
                    pl.when(pl.col("roi") == roi)
                    .then(pl.col(f"mean_intensity_{intensity_type}_temp"))
                    .otherwise(pl.col(f"mean_intensity_{intensity_type}"))
                    .alias(f"mean_intensity_{intensity_type}"),
                    
                    pl.when(pl.col("roi") == roi)
                    .then(pl.col(f"max_intensity_{intensity_type}_temp"))
                    .otherwise(pl.col(f"max_intensity_{intensity_type}"))
                    .alias(f"max_intensity_{intensity_type}"),
                    
                    pl.when(pl.col("roi") == roi)
                    .then(pl.col(f"min_intensity_{intensity_type}_temp"))
                    .otherwise(pl.col(f"min_intensity_{intensity_type}"))
                    .alias(f"min_intensity_{intensity_type}")
                ])
                .drop([f"mean_intensity_{intensity_type}_temp", 
                       f"max_intensity_{intensity_type}_temp", 
                       f"min_intensity_{intensity_type}_temp"])
            )
            
            print(f"  Updated {intensity_type} intensity for ROI {roi}")
    else:
        print(f"  No intensity data available for ROI {roi}")

print(f"\nFinal polygons shape: {polygons.shape}")
print(f"Final polygons columns: {polygons.columns}")

# Check that we have intensity data for both ROIs
for roi in rois:
    roi_data = polygons.filter(pl.col("roi") == roi)
    for intensity_type in intensity_types:
        col_name = f"mean_intensity_{intensity_type}"
        non_null_count = roi_data.filter(pl.col(col_name).is_not_null()).shape[0]
        total_count = roi_data.shape[0]
        print(f"ROI {roi}, {intensity_type}: {non_null_count}/{total_count} cells have intensity data")

polygons, roi_offsets = arrange_rois(polygons, max_columns=2, padding=100)


# Modified weighted centroids calculation to include all intensity types
agg_expressions = [
    pl.col("area").sum().alias("area"),
    ((pl.col("centroid_x") * pl.col("area")).sum() / pl.col("area").sum()).alias("x"),
    ((pl.col("centroid_y") * pl.col("area")).sum() / pl.col("area").sum()).alias("y"),
    ((pl.col("z").cast(pl.Float64) * pl.col("area")).sum() / pl.col("area").sum()).alias("z"),
    pl.col("roi").first(),
]

# Add aggregation expressions for each intensity type
for intensity_type in intensity_types:
    mean_col = f"mean_intensity_{intensity_type}"
    max_col = f"max_intensity_{intensity_type}"
    min_col = f"min_intensity_{intensity_type}"
    
    # Check if columns exist before adding aggregation
    if mean_col in polygons.columns:
        agg_expressions.extend([
            ((pl.col(mean_col) * pl.col("area")).sum() / pl.col("area").sum()).alias(f"mean_intensity_{intensity_type}"),
            pl.col(max_col).max().alias(f"max_intensity_{intensity_type}"),
            pl.col(min_col).min().alias(f"min_intensity_{intensity_type}"),
        ])

weighted_centroids = (
    polygons.group_by(pl.col("roilabel"))
    .agg(agg_expressions)
    .sort("roilabel")
)

# %%
weighted_centroids = weighted_centroids.to_pandas().set_index("roilabel")
weighted_centroids.index = weighted_centroids.index.astype(str)

# %%
# ident = df.join(spots[["index", "target"]], left_on="spot_id", right_on="index", how="left")
# %%
molten = df.group_by([pl.col("roilabel"), pl.col("target")]).agg(pl.len())

# %%
cbg = (
    molten.with_columns(
        gene_name=pl.when(
            pl.col("target").str.contains(r"-2\d+").and_(~pl.col("target").str.starts_with("Blank"))
        )
        .then(pl.col("target").str.split("-").list.get(0))
        .otherwise(pl.col("target"))
    )
    .drop("target")
    .pivot("gene_name", index="roilabel", values="len")
    .fill_null(0)
    .sort("roilabel")
    .to_pandas()
    .set_index("roilabel")
)
cbg.index = cbg.index.astype(str)

# %%

import anndata as ad
import cmocean  # colormap, do not remove
import colorcet as cc  # colormap, do not remove
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import spaco
import tifffile
from shapely import MultiPolygon, Point, Polygon, STRtree

adata = ad.AnnData(cbg)
adata.obs = weighted_centroids.reindex(adata.obs.index)

n_genes = adata.shape[1]
sc.pp.calculate_qc_metrics(
    adata, inplace=True, percent_top=(n_genes // 10, n_genes // 5, n_genes // 2, n_genes)
)
#sc.pp.filter_cells(adata, min_counts=10)
#sc.pp.filter_cells(adata, max_counts=1200)
#sc.pp.filter_genes(adata, min_cells=3)
# adata = adata[(adata.obs["y"] < 23189.657075200797) | (adata.obs["y"] > 46211.58630310604)]
adata.obsm["spatial"] = adata.obs[["x", "y"]].to_numpy()
adata = adata[adata.obs["area"] > 500]
ok = np.ones(adata.shape[0], dtype=bool)
# ok[baddies] = False
adata = adata[ok]
sc.pl.violin(
    adata,
    ["n_genes_by_counts", "total_counts"],
    jitter=0.4,
    multi_panel=True,
)
print(np.median(adata.obs["total_counts"]), len(adata))

# adata.write_h5ad(path / "segmentation_counts.h5ad")
# %%
plt.scatter(adata.obs["x"][::1], adata.obs["y"][::1], s=1, alpha=0.3)


# %%

#
# if not "raw" in adata.layers:
#    adata.layers["raw"] = adata.X.copy()
#    #adata, plot = normalize_total(adata)
#    adata.obs["total_intensity"] = adata.obs["mean_intensity"] * adata.obs["area"]
#    adata.obs["log_total_intensity"] = np.log10(adata.obs["total_intensity"] + 1)


# %%
# Save intermediate results
# Save results combined and separate
output_h5ad_combined = output_dir / "combined_rois_processed.h5ad"
adata.write_h5ad(output_h5ad_combined)
print(f"Combined AnnData saved to: {output_h5ad_combined}")

# Save separate AnnData files per ROI
print("\nSaving separate AnnData files per ROI:")
for roi in adata.obs['roi'].unique():
    print(roi)
    # Subset to this ROI only
    roi_mask = adata.obs['roi'] == roi
    adata_roi = adata[roi_mask, :].copy()
    
    # Filter genes that have no expression in this ROI
    gene_counts = adata_roi.X.sum(axis=0)
    if hasattr(gene_counts, 'A1'):  # sparse matrix
        gene_counts = gene_counts.A1
    expressed_genes = gene_counts > 0
    print(sum(gene_counts == 0), "zero expressing genes")
    adata_roi = adata_roi[:, expressed_genes].copy()
    
    # Save ROI-specific file
    roi_output_file = output_dir / f"{roi}_processed.h5ad"
    adata_roi.write_h5ad(roi_output_file)
    print(f"  {roi}: {adata_roi.n_obs} cells, {adata_roi.n_vars} genes -> {roi_output_file}")

print(f"\nFiles saved:")
print(f"  Combined: {output_h5ad_combined}")
for roi in adata.obs['roi'].unique():
    print(f"  {roi}: {output_dir / f'{roi}_processed.h5ad'}")