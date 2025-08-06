# Getting Started

This repository includes two example notebooks located in the `notebooks` folder:

- `eccc_downscaling_finetune.ipynb`
- `eccc_downscaling_inference.ipynb`

These notebooks demonstrate how to use the **Prithvi Weather Foundation Model** for a downscaling task on Canada’s operational Numerical Weather Prediction (NWP) systems. Specifically, the goal is to downscale forecasts from the **Global Deterministic Prediction System (GDPS)**—which provides 10-day forecasts at ~15 km resolution—to the **High-Resolution Deterministic Prediction System (HRDPS)**, which produces 48-hour forecasts at ~2.5 km resolution.

For more information about the dataset, see our preliminary work using GANs: [arxiv.org/pdf/2412.06958](https://arxiv.org/pdf/2412.06958)

The notebooks walk through the full pipeline using a single GDPS-HRDPS data pair, covering preprocessing, training, and inference. We also provide links to the full dataset and configuration files so you can run your own experiments.

---

# Dataset Description

### GDPS Inputs

| Variable         | Description                                                  | Units |
|------------------|--------------------------------------------------------------|--------|
| U<sub>surf</sub> | Zonal wind at surface (10 m)                                 | m/s    |
| V<sub>surf</sub> | Meridional wind at surface (10 m)                            | m/s    |
| T<sub>surf</sub> | Air temperature at surface (1.5 m)                           | °C     |
| T<sub>546</sub>  | Air temperature at 546 hPa (interpolated)                    | °C     |
| U<sub>546</sub>  | Zonal wind at 546 hPa                                        | m/s    |
| V<sub>546</sub>  | Meridional wind at 546 hPa                                   | m/s    |
| W<sub>546</sub>  | Vertical motion at 546 hPa (interpolated)                    | Pa/s   |

### HRDPS Targets

| Variable | Description                        | Units |
|----------|------------------------------------|--------|
| u10      | Zonal wind at surface (10 m)       | m/s    |
| v10      | Meridional wind at surface (10 m)  | m/s    |

### HRDPS Static Covariates

| Variable | Description        | Units     |
|----------|--------------------|-----------|
| me       | Orography          | m         |
| mg       | Water/land mask    | fraction  |
| z0       | Surface roughness  | m         |

---

# The Task

The objective is to downscale GDPS outputs by a factor of 8 to match the spatial resolution of HRDPS using the **Prithvi Weather Foundation Model**. The following data pipeline is used for preprocessing.

---

## Data Pipeline

- **Regridding**: GDPS and HRDPS are provided in different rotated coordinate systems. We regrid GDPS to match the HRDPS grid using nearest-neighbor interpolation.
- **Downsampling**: The regridded GDPS is downsampled by a factor of 8 to approximate a 20 km spatial resolution.
- **Scaler Computation**: Compute normalization scalers (mean and standard deviation) from the training partition.
- **Data Sampling**: Each GDPS-HRDPS pair is large (~1.5 GB), so we sample random spatial crops to form an intermediate dataset for training.
- **Static Covariates**: Static features (e.g., orography, roughness) are used to aid model learning.

---

## Model

We use a UNet architecture, integrating the **Prithvi encoder** as a deep feature extractor.

---

# Running Training and Inference

You can download all required files from our [Hugging Face repository](https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling).

We provide:

- **Configuration files**: YAML files containing experiment settings.
- **Sample data**: One preprocessed GDPS-HRDPS pair for demonstration.
- **Scalars**: Precomputed normalization statistics.
- **Indices**: JSON files mapping file paths for input/target data pairs.
- **Pretrained weights**: A model checkpoints.

# Step-by-step guide to setup the enviroment for running the notebooks
## Clone this repository and the Pritvhi weather foundation model:
```bash
git clone https://github.com/NASA-IMPACT/Prithvi-WxC
git clone https://github.com/IBM/granite-wxc.git
cd Prithvi-WxC
pip install '.[examples]'
cd ../granite-wxc
pip install '.[examples]'
```
## Clone Hugging Face Repository

```bash
cd granite-wxc/
git clone https://huggingface.co/ibm-granite/granite-geospatial-wxc-downscaling
```

## Setup Virtual Enviroment
Using uv
```bash
uv venv  .venv
source .venv/bin/activate
```

Once cloned, you can run the example notebooks. 
The next section shows the case of using the full dataset.

---

# Using the Full Dataset

The Hugging Face repository includes only a sample. Download the full datasets from the following links:

- **HRDPS**: [Download](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/hrdps/)
- **GDPS**: [Download](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/gdps_regridded/)
- **Static data**: [Download](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/geophy.nc)

Before training, you must regrid the GDPS data to match the HRDPS domain.

---

### Data Preprocessing

Use `preprocess.py` to interpolate GDPS onto the HRDPS grid using nearest-neighbor interpolation. The regridded data is then downsampled by a factor of 8.

![Regridding](regrid.png)
*Figure 1: Example of zonal and meridional winds on GDPS and HRDPS grids. HRDPS domain is shown by a black rectangle.*


> ***Obs***: Once regridding is complete, the GDPS data matches the HRDPS resolution. The Dataset class then handles downsampling automatically by a factor of eight, see the implementation in [eccc.py](https://github.com/victor-nasc/granite-wxc/blob/main/granitewxc/datasets/eccc.py)


---

### Index Files

You must define index files in JSON format for both input and static data.

**Data Pair Indices:**

```json
{
  "0": ["<GDPS_file.nc>", "<HRDPS_file.nc>"],
  "1": ["<GDPS_file.nc>", "<HRDPS_file.nc>"]
}
```

**Static Covariate Indices:**

```json
{
  "static_regridded_gdps": "<Static_GDPS_file.nc>",
  "static_hrdps": "<Static_HRDPS_file.nc>"
}
```

---

### Scaler Computation

Compute normalization statistics with:

```bash
python3 compute_scalars.py --config_path <CONFIG> --save_dir <DIR>
```

> **Note:** These statistics should be computed using only the training data.

---
### Training

To train with your own dataset, refer to `eccc_downscaling_finetune.ipynb` for an example setup. You will likely need to adapt the code into your own training script to train beyond the example.


