# Getting started

You can download all the essential files to run the examples via our [Hugging Face](https://huggingface.co/victornasc/granite-geospatial-eccc-downscaling) repository

We provide the following:
- <ins>Configuration files</ins>: `YAML` files containing settings related to the model, data, file paths, and more
- <ins>Data sample</ins>: This example uses only one preprocessed data sample. If you'd like to train with more data, see the section below
- <ins>Scalars</ins>: These files contain the mean and standard deviation statistics used to normalize the data before feeding it to the model
- <ins>Indices</ins>: Data is organized in `JSON` files mapping unique string indices to pairs of file paths (low resolution, high resolution). The Dataset class uses these indices to match the files correctly
- <ins>Pretrained weights</ins>: These are our best checkpoints from training using the ECCC data

To reproduce the outputs shown in the example notebooks, download the required files and run the notebooks

```bash
git clone https://huggingface.co/victornasc/granite-geospatial-eccc-downscaling ./experiments
```

# Finetuning with the full dataset 

### Downloading

On [Hugging Face](https://huggingface.co/victornasc/granite-geospatial-eccc-downscaling), we provide only a sample of the data. You can download the remaining HRDPS and GDPS  datasets from the following links:

- HRDPS (High-Resolution Deterministic Prediction System): [Download Here](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/hrdps/)

- GDPS (Global Deterministic Prediction System): [Download Here](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/gdps_regridded/)

- Static Data: [Download Here](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/geophy.nc)

<div align="center">
  
### GDPS Predictors

| Variable    | Description                                                                                     | Units   |
|-------------|-------------------------------------------------------------------------------------------------|---------|
| U<sub>surf</sub>  | True geographical West-East (zonal) component of the horizontal wind at the surface (10 m)     | [m/s]   |
| V<sub>surf</sub>  | True geographical South-North (meridional) component of the horizontal wind at the surface (10 m) | [m/s]   |
| T<sub>surf</sub>  | Air temperature at the surface (1.5 m)                                                        | [°C]    |
| T<sub>546</sub>   | Air temperature vertically interpolated at 546 hPa                                            | [°C]    |
| U<sub>546</sub>   | True geographical West-East (zonal) component of the horizontal wind at 546 hPa               | [m/s]   |
| V<sub>546</sub>   | True geographical South-North (meridional) component of the horizontal wind at 546 hPa        | [m/s]   |
| W<sub>546</sub>   | Vertical motion vertically interpolated at 546 hPa                                            | [Pa/s]  |

### HRDPS Predictands

| Variable | Description                                                                                   | Units |
|----------|-----------------------------------------------------------------------------------------------|--------|
| u10      | True geographical West-East (zonal) component of the horizontal wind at the surface (10 m)   | [m/s]  |
| v10      | True geographical South-North (meridional) component of the horizontal wind at the surface (10 m) | [m/s]  |

### HRDPS Static Covariates

| Variable | Description              | Units       |
|----------|--------------------------|-------------|
| me       | Model orography          | [m]         |
| mg       | Water/land mask          | [fraction]  |
| z0       | Roughness length         | [m]         |

</div>

### Data preprocessing 

The map projection of the GDPS outputs is a Yin-Yang grid with a different rotated latitude-longitude map projection than the one of the HRDPS grid. To align the grids, we project the GDPS grid to the HRDPS grid using nearest neighbor interpolation. The interpolated data is then reduced by a factor of 8 (20-km nominal resolution) as the input of the AI downscaling method.

<p align="center">
   <img src=regrid.png alt="ECCC data regridding" width="70%"/>
   <br><em>
      Figure 1: Example of zonal and meridional winds on the GDPS and HRDPS grids<br>
      Note that the GDPS grid was cropped over North America<br>
      The domain of HRDPS grid is indicated by a black rectangle
   </em>
</p>

We provide the `preprocess.py` script to make the reggridding. Make sure that the HRDPS has exactly 8× the spatial resolution of the GDPS. You may need to crop additional rows and columns to ensure this alignment

### Scalars 
We provide a [scalar generator](compute_scalars.py) script to compute the statistics used for normalizing the data before feeding it into the model

```bash
python3 compute_scalars.py --config_path <CONFIG> --save_dir <DIR>
```

**Important:** These statistics should be computed exclusively using your **training data**!

### Data indexes

The data is organized in `JSON` files, mapping unique indices (as string keys) to pairs of file paths. Each index key points to an array containing two strings:

````
{
  "0": ["<GDPS data file>", "<HRDPS data file>"],
  "1": ["<GDPS data file>", "<HRDPS data file>"],
  ...
}
````

Each entry represents a paired dataset where the GDPS and HRDPS files correspond to the same timestamp and region

We provide an [index generator](index_generator.py) to help you generate your JSON index file automatically

```bash
python3 index_generator.py --gdps_paths <GDPS> --hrdps_paths <HRDPS> --output <OUTPUT>
```

The static data should have an index file as well

````
{
  "static_regridded_gdps": "<Static GDPS file>", "static_hrdps": "<Static HRDPS file>"
}
````

After setting up all the necessary requirements for your additional data, you should be able to run training and inference
