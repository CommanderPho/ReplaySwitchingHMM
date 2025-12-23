# Drift-diffusion dynamics of hippocampal replay

[![Paper](https://img.shields.io/badge/paper-PDF-blue)](https://doi.org/10.1101/2025.10.14.682470)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2025.10.14.682470-blue)](https://doi.org/10.1101/2025.10.14.682470)


Official code repository for "Drift-diffusion dynamics of hippocampal replay" by Zhongxuan Wu and Xue-Xin Wei.

---

## Repository structure

```text
.
├── ssm/                                    # Implementation of the switching HMM used in the paper
├── submission/                             # Code to reproduce paper figures (to be reorganized during revision)
├── preprocessing_utils.py                  # Data preprocessing utilities
├── preprocessing_Pfeiffer1D.py             # Data preprocessing scipt
├── train_Pfeiffer1D.py                     # Probabilistic inference on the preprocessed data
├── data_train_shuffle.py                   # Time-bin + neuron-identity shuffles + runs inference
├── data_train_shuffle_pfs.py               # Place-field rotation shuffle + runs inference
├── data_train_simulation_recovery.py       # Simulated SWRs (drift-diffusion only) for parameter recovery + runs inference
└── data_train_simulation_full_recovery.py  # Simulated SWRs (all basic dynamics) for parameter recovery + runs inference
```


---

## Data availability

This study uses data from [Pfeiffer and Foster (2015)](https://www.science.org/doi/full/10.1126/science.aaa9633). Please contact Brad Pfeiffer and David Foster to access the analyzed datasets.

---