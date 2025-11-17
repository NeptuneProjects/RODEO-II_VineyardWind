# RODEO II - Vineyard Wind Data Analysis Workflow

William Jenkins, Ph.D.  
Scripps Institution of Oceanography
University of California San Diego

This repository depends on the [`rodeo`](https://github.com/NeptuneProjects/RODEO-II) Python package.

## Workflow

### Configuration

This workflow makes use of a number of configuration files, data sources, etc.
Additionally, the outputs are numerous and varied.
Paths are specified in a centralized manner to make it easy to change the location of data files, outputs, etc., and can be found at `config/paths.toml`

Since the workflow is designed to be run either locally or on a remote server, computer-specific paths can be set in the `paths.toml` file.
To set this up, run the configuration script and specify an alias for the computer you're on:
```bash
python src/vineyard/config.py set-id --alias=<your_alias>
```
The script will either:
- Create a new `.env` file in the root directory with your computer identifier and alias if one does not exist.
- Update the existing `.env` file with your computer identifier and alias if one already exists and the ID and alias are present.
- Append your computer identifier and alias to the existing `.env` file if one already exists and the ID and alias are not present.

To update the alias, run:
```bash
python src/vineyard/config.py set-alias <new_alias>
```

Running the script will also show what paths are set for your computer and whether or not they exist. To view the paths set for your computer, run:
```bash
python src/vineyard/config.py
```

With the alias set, the paths in `config/paths.toml` will be updated to point to the correct locations for your computer.
Ensure the alias is consistent with the headings in the TOML file.
For example, for three aliases, `work-laptop`, `home-desktop`, and `lab-server`, the paths file could look like this:
```toml
[computer.work-laptop]
data_dir = "D:/Projects/BigData"
output_dir = "E:/Results"

[computer.home-desktop]
data_dir = "C:/Users/me/Datasets"
output_dir = "D:/ProjectOutput"

[computer.lab-server]
data_dir = "/data/shared/datasets"
output_dir = "/home/user/results"
temp_dir = "/tmp/myproject"
```

If any keys in the computer-specific sections are the same as the default keys, the default values will be overridden.

### Construct an inventory of data files.

1. Configure the dataset inventory using the file `config/inventory.toml`.
2. Build the dataset inventory by running:  
   `python scripts/setup/inventory.py`  
   The inventory is performed for each station, and saved as:  
   `data/inventory/inventory_SHRU905_VLA1.csv`
3. Detect pile driving strikes in the acoustic data by running:
   `python scripts/process/strikes_find.py`
4. Extract, process, and save all strikes to an HDF5 database:  
   `python scripts/process/strikes_save.py`
5. Compute cross-correlations between all strike pairs:  
   `python scripts/process/strikes_corr.py`
6. Build database of templates:
   `python scripts/process/template_extraction.py`
   