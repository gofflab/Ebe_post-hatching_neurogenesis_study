# code for experiment EBE00216_s04
# 24 hour hatchling pulse, 4 day chase
# rois: bottom
#!/bin/bash

conda activate fishtools

#  deconvolve in experiment folder
cp -r ../basic/ .
preprocess deconv batch . --basic-name=all

cd analysis/deconv/

# ebe_devprobeset_targets - registration and spot calling
preprocess deconv compute-range . --overwrite 
preprocess register batch . --config=config.json --codebook=/working/fishtools/2025-05/berry/ebe_devprobeset_targets.json --threads=15
preprocess spots optimize . --codebook=/working/fishtools/2025-05/berry/ebe_devprobeset_targets.json --rounds=8 --threads=15
preprocess spots batch . --codebook=/working/fishtools/2025-05/berry/ebe_devprobeset_targets.json --threads=8 --split
preprocess stitch register . --idx=1 --max-proj --codebook=ebe_devprobeset_targets --debug --overwrite
preprocess spots stitch . --codebook=/working/fishtools/2025-05/berry/ebe_devprobeset_targets.json --threads=15

## ebe_devprobeset_targets - check shifts, threshold, spots
preprocess check-shifts . --codebook=codebooks/ebe_devprobeset_targets.json
preprocess spots threshold . --codebook=codebooks/ebe_devprobeset_targets.json
### threshold input: 3

# ebe_tricycle_targets registration and spot calling

## ebe_tricycle_targets - registration and spot calling
preprocess deconv compute-range . --overwrite
preprocess register batch . --config=config.json --codebook=/working/fishtools/2025-05/berry/ebe_tricycle_targets.json --threads=15
preprocess spots optimize . --codebook=/working/fishtools/2025-05/berry/ebe_tricycle_targets.json --rounds=8 --threads=15
preprocess spots batch . --codebook=/working/fishtools/2025-05/berry/ebe_tricycle_targets.json --threads=8 --split
preprocess stitch register . --idx=1 --max-proj --codebook= --debug --overwrite
preprocess spots stitch . --codebook=/working/fishtools/2025-05/berry/ebe_tricycle_targets.json --threads=15

## ebe_tricycle_targets - check shifts, threshold, spots
preprocess check-shifts . --codebook=codebooks/ebe_tricycle_targets.json
preprocess spots threshold . --codebook=codebooks/ebe_tricycle_targets.json
### threshold input: 3

## reddot segmentation registration - initial registration
preprocess register batch . --codebook=edu.json --config=config.json --threads=2
preprocess stitch register . --idx=0 --max-proj --codebook=reddot --overwrite
preprocess check-shifts . --codebook=codebooks/reddot.json

## reddot -  check shifts
preprocess stitch register . --idx=0 --max-proj --codebook=reddot --overwrite
preprocess check-shifts . --codebook=codebooks/reddot.json

## fuse zarr
preprocess stitch fuse . --codebook=reddot --overwrite

## segmentation
python /working/fishtools/segmentation/distributed/distributed_segmentation.py . --channels="af,reddot"

## overlay edu
python /working/fishtools/segmentation/distributed/overlay_intensity.py . --channel="edu" --intensity-name="fused.zarr"

## overlay reddot
python /working/fishtools/segmentation/distributed/overlay_intensity.py . --channel="reddot" --intensity-name="fused.zarr"

## create chunks
python /working/fishtools/working/fishtools/fishtools/preprocess/spots/overlay_spots.py .  --codebook=ebe_devprobeset_targets   --seg-codebook=edu   --overwrite
python /working/fishtools/working/fishtools/fishtools/preprocess/spots/overlay_spots.py . --codebook=ebe_tricycle_targets --seg-codebook=edu --overwrite

## create anndata
### use custom script
