## Downloading ECCC data

You can download the HRDPS and GDPS datasets, which correspond to high-resolution and low-resolution data respectively, from the following links:

HRDPS (High-Resolution Deterministic Prediction System): [Download Here](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/hrdps/)

GDPS (Global Deterministic Prediction System): [Download Here](https://hpfx.collab.science.gc.ca/~snow000/hrdps_domain/gdps_regridded/)

## Getting started

The data is organized in `JSON` files, mapping unique indices (as string keys) to pairs of file paths. Each index key points to an array containing two strings:

````
{
  "index": ["GDPS data path", "HRDPS data path"],
  ...
}
````

Each entry represents a paired dataset where the GDPS and HRDPS files correspond to the same timestamp and region. The data files are stored in **NetCDF** format (`.nc`)

We provide an [index generator](examples/eccc_downscaling/index_generator.py) to help you generate your JSON index file automatically

```bash
python3 index_generator.py --gdps_paths <GDPS> --hrdps_paths <HRDPS> --output <OUTPUT>
```

Additionally, we provide a [scalar generator](examples/eccc_downscaling/compute_scalars.py), which is used for normalizing the data before feeding it into the model

**Important:** These statistics should be computed exclusively using your **training data**!


```bash
python3 compute_scalars.py --config_path <CONFIG> --save_dir <SAVE>
```

## Configuration Files

Configuration files contain settings related to the model, data, and file paths

Once the data is downloaded and the indexes and scalars are generated, you need to update the file paths in the `YAML` configuration files located at `/examples/eccc_downscaling/`