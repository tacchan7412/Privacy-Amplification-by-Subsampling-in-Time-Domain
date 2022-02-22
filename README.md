# Privacy Amplification by Subsampling in Time Domain

Official implementation of the paper:
Privacy Amplification by Subsampling in Time Domain, Koga, T., Meehan, C. & Chaudhuri, K. (25th International Conference on Artificial Intelligence and Statistics (AISTATS), Virtual, Mar. 2022)
[[arxiv]](https://arxiv.org/abs/2201.04762)

## Dependencies
- Python (3.9.0)

## How to Run Experiments
1. Prepare datasets
    1. PeMS
    Visit [PeMS website](https://pems.dot.ca.gov/) and register.
    Then, from "Data Clearinghouse" section, download available files (e.g., d04_text_station_5min_2017_01_01.txt.gz) by specifying Type as "Station 5-Minute" and District as "District 4" and selecting Jan. 2017 as starting data period.
    Put all files under `data/PEMS/pems-bay-raw`.
    Run `python generate_flow_df.py` to get `pems-bay-flow.h5` (put it under `data/PEMS`).
    2. Gowalla
    Visit [website](https://snap.stanford.edu/data/loc-Gowalla.html) and download `loc-gowalla_totalCheckins.txt.gz`.
    Place it under `data/Gowalla`.
    3. Foursquare
    Visit [website](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) and download Global-scale Check-in Dataset.
    Place all files under `data/Foursquare`.

2. Install libraries
```
pip install -r requirement.txt
```

3. Run Notebooks
- Table 1: `I_validation.ipynb`
- Table 2: `real_data_exp.ipynb`
- Figure 1, 2 and Figure 1 in Appendix: `synth_data_exp.ipynb`
- Figure 2 in Appendix: `random_ssf_numerical.ipynb`
