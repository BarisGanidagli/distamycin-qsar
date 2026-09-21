# Distamycin Analogue QSAR Pipeline

Baris Ganidagli and David E. Thurston, King's College London

A small data QSAR study of the cytotoxicity of distamycin analogues against L1210 murine leukaemia cells. Running the two notebooks in order reconstructs the analysis from the raw input data and reproduces all figures (except for Figures 1, 2 and 6, which are structure figures), tables and predictions presented in the manuscript.

## Layout

The `data/raw` folder consists of the three source files used throughout the project, including the fragment definitions and measured activity values. The `data/processed` folder contains the datasets derived from these files, together with three cached ChEMBL records. The `modules` folder contains the Python package imported by the notebooks. The `results` and `saved_models` folders consist of generated outputs that are recreated each time the notebooks are run.

The modelling dataset is `data/processed/processed_data.csv` and it contains 2,651 unique structures, including 116 labelled and 2,535 unlabelled. These structures were obtained from an enumerated library of 2,640 compounds, along with additional compounds identified from the literature.

## Notebooks

`library_generation.ipynb` generates the combinatorial library, adds the literature compounds, removes duplicate canonical SMILES, and produces `cleandata.csv` and `processed_data.csv`.

`distamycin_sar_modelling.ipynb` contains the modelling pipeline and generates the Supporting Information for the manuscript. It trains four model families using scaffold-based cross-validation, fits a Free-Wilson model to 90 of the labelled compounds, and evaluates the models using independent ChEMBL actives. No unlabelled compounds are selected for further analysis because the external validation shows that the predicted range is too narrow to reliably distinguish between compounds in this series.

## Running

Developed and tested on Python 3.13.6.

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `library_generation.ipynb` first, then `distamycin_sar_modelling.ipynb`. For both notebooks, use **Restart Kernel and Run All Cells**. Several cells redefine variables used by later cells, so running the notebooks partially or out of order may produce results that do not correspond to the code above them. The modelling notebook takes approximately 60 minutes to run on an Apple M4 MacBook Air because `n_jobs=-1` is used throughout.

The notebooks locate the project directory by searching upwards for `requirements.txt`, `.git` or `README.md`. **The folder structure must therefore be preserved**. Extracting the archive without retaining the directory structure will cause the data and module paths to fail.

## Reproducibility

`SEED = 42` is defined at the top of the modelling notebook and is used for Random Forest, Gaussian Process restarts, the XGBoost subsampling and scaffold fold assignment. Bootstrap resampling in Sections 8.1, 9.6 and 9.7 uses a separate `default_rng`, ensuring that it does not affect the global random state.

Cached ChEMBL records are included in the repository, so the project can be rebuilt without network access. Deleting these files will trigger a live query, which may return different records if the ChEMBL database has been updated since 10 August 2026.

## Data sources

The activity data were obtained from the combinatorial distamycin library reported by Boger and colleagues:

- Boger, D.L., Fink, B.E. and Hedrick, M.P. (2000) *J. Am. Chem. Soc.* **122**, 6382–6394.
- Boger, D.L., Dechantsreiter, M.A., Ishii, T., Fink, B.E. and Hedrick, M.P. (2000) *Bioorg. Med. Chem.* **8**, 2049–2057.

External validation compounds were retrieved from ChEMBL target CHEMBL386 (L1210 murine leukaemia) on 10 August 2026.

## Citation

Please cite the manuscript and this repository: 10.5281/zenodo.22882946

## License

The code is available under the MIT license as specified in `LICENSE`. The activity data were obtained from the published studies by Boger and colleagues cited above. This attribution should be retained when reusing the dataset.