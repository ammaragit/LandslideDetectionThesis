# Reliability-Aware Multimodal Landslide Detection (HAEFNet)

Official PyTorch implementation of “Reliability Aware Multimodal Detection of Active Landslides via Evidence Theoretic Fusion of InSAR, Optical, and DEM Data.” The framework fuses InSAR, optical, and DEM cues for pixel-wise landslide mapping with reliability indicators.

## Repository Layout
- Entry points: `train.py` and `predict.py`.
- Configs: `configs/exp_haefnet.yml` (set data paths and modality flags).
- Modules: `models/` (fusion blocks, encoders, heads) and `utils/` (dataset loader, augmentations, losses, optimizer, logging, checkpoints).
- Datasets and model weights are not tracked in Git. Use `data/` for datasets, `pretrained/` for backbone weights, and `work_dir/` or `--output_dir` for experiment outputs.

## Requirements and Installation
1. Install Miniconda or Anaconda with CUDA-enabled PyTorch support.
2. Create the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate haefnet
   ```
   The environment file specifies Python 3.9 and MintPy. Install a PyTorch build compatible with your CUDA setup and any additional packages required by the scripts you plan to run.
3. Ensure `gdal` and `rasterio` can read your GeoTIFFs when using those workflows. Set `CUDA_VISIBLE_DEVICES` if needed.

## Data Preparation
Datasets and model weights are distributed separately and are not included in this repository. Obtain them from an authorized source, then place them in the paths configured in `configs/exp_haefnet.yml`. Set `root_dir` to the dataset location on your machine; avoid committing machine-specific absolute paths.

## InSAR Preprocessing (ASF + MintPy)
1. Discover and download SAR stacks from ASF DAAC (https://search.asf.alaska.edu/#/) for your area of interest.
2. Process interferograms and time series with MintPy (https://github.com/insarlab/MintPy): unwrap, geocode, and export LOS displacement/coherence to GeoTIFFs.
3. Resample outputs to the optical/DEM grid used by the dataset; store them under the InSAR modality folders referenced in your split lists. Record processing parameters for reproducibility.

## Training
- Edit `configs/exp_haefnet.yml` to set paths, batch size, learning rate, and modality toggles.
- Launch training:
  ```bash
  python train.py --config configs/exp_haefnet.yml --name exp_haefnet
  ```
  Checkpoints are saved under `checkpoints/<timestamp>_<name>/` by default.

## Inference and Visualization
Run prediction and diagnostics:
```bash
python predict.py --config configs/exp_haefnet.yml \
  --model_path <checkpoint_path> \
  --output_dir work_dir/demo_pred
```
This generates per-pixel maps, RGB overlays, probability maps, and uncertainty visualizations.

## Code Availability
The source code for this study is available in this repository under its license. Data and model weights should be obtained from an authorized source. InSAR preprocessing relies on ASF data access and the open-source MintPy toolkit.
