# Image Preprocessing

This is instructions for image preprocessing to adata matrix for further analysis.

## Deconvolution

Prior to deconvolution, a basic directory needs to be copies to the working directory of the experiment. The basic file is applies a corrective field for each laser line and is specific for each microscope. Deconvolution command requires a GPU. This command outputs a subdirectory analysis/deconv/ which is the working directory for further preprocessing.

```sh
preprocess deconv batch . --basic-name=all
```

## Codebook Registration and Spot Calling

Registration and spot calling need to be done for each codebook. The config.json file can be created with empty brackets, but can have information for flags. Once a coodebook is used for registration, it is copied to a created codebook subdirectory.

```sh
preprocess deconv compute-range . --overwrite
preprocess register batch . --config=config.json --codebook=/path/to/codebook.json --threads=15
preprocess spots optimize . --codebook=/path/to/codebook.json --rounds=8 --threads=15
preprocess spots batch . --codebook=/path/to/codebook.json --threads=8 --split
preprocess stitch register . --idx=1 --max-proj --codebook=codebook --debug --overwrite
preprocess spots stitch . --codebook=/path/to/codebook.json --threads=15
```

## Non-bit Registration

Non-bit registration such as cell boundaries stains only require registration.

```sh
preprocess register batch . --codebook=/path/to/codebook.json --config=config.json --threads=2
preprocess stitch register . --idx=0 --max-proj --codebook=codebook 
```

## Checking Shifts and Thresholding

The check-shifts create an output subdirectory in analysis to verify tiels are registered correctly. There are a multiple outputs but primarily shift_layouts and L2 distance can be used to assess registration against the reference. Thresholding can used to assess distribution of spots. Each codebook needs to be ran separately. Thresholding is only for panel codebooks.

```sh
preprocess check-shifts . --codebook=codebooks/codebook.json
preprocess spots threshold . --codebook=codebooks/codebook.json
```

## Registration Troubleshooting

After check-shifts, there may be tiles missing or having major shifts from the reference (L2 distance greater than 10). Use check-shifts throughout correction to check tile improvement. The flags --threshold and --fwhm can be changed to improve registration. These values can range from 2-10 depending on the quantity and quality of fidicual spots. After registration correction, spot calling needs to run again with --overwrite flag for bit codebook panels. To check status and overwriting of command series, use preprocess status command.

```sh
# missing tiles
preprocess register run . tile --roi=roi --codebook=codebooks/codebook.json --debug --config=config.json --threshold=2 --fwhm=2
preprocess stitch register . roi --idx=1 --max-proj --codebook=codebook --debug --overwrite
preprocess check-shifts . --codebook=codebooks/codebook.json

# tile correction
preprocess register batch . --config=config.json --codebook=codebooks/codebook.json --overwrite --use-brightest=10 --offset-brightest=1 --only-median-gt=10 --fwhm=2 --threshold=2 --ref=2_10_18 --threads=2 --debug
preprocess check-shifts . --codebook=codebooks/codebook.json

# spot calling correction
preprocess spots batch . --codebook=codebooks/codebook.json --threads=8 --split --overwrite
preprocess stitch register . roi --idx=1 --max-proj --codebook=codebook --debug --overwrite
preprocess spots stitch . --codebook=codebooks/codebook.json --threads=8 --overwrite

# check progress
preprocess status . 
```

## Fuse Zarr

Use the non-bit (segmentation) codebook to create a zarr.

```sh
reprocess stitch fuse . --codebook=codebook/codebook.json --overwrite
```

## Segmentation

A cellpose model should already be trained and assessable. Edit the config.json file to add path of cellpose model with flags for running the command. Z index is for the center most z.

```sh
preprocess stitch n4 . --codebook=codebook --z-index=18
segment batch . --codebook=codebook
```

## Overlay Stains

Overlay function can be used to add intentsity of stains such as EdU fluorescence. Multiple panel codebooks can be used.

```sh
segment overlay all . --codebook=codebook1 --codebook=codebook2 --seg-codebook=reddot --intensity-codebook=edu --segmentation-name=output_segmentation.zarr
```

## Export Adata Matrix

```sh
segment export . --codebook=codebook1 --codebook=codebook2 --seg-codebook=reddot --segmentation-name=output_segmentation.zarr
```
