# CardioCXR

This repository contains scripts to process and model ECG data for predictive tasks. Follow the steps below to replicate the experiments.

# 1. Download Required Datasets

Before running the scripts, download the following files and place them in the root directory of this repository:

mimic-cxr-2.0.0-metadata.csv.gz from https://physionet.org/content/mimic-cxr/2.0.0/

cxr-lt-2024/labels.csv from https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/

machine_measurements.csv from https://physionet.org/content/mimic-iv-ecg/1.0/

mimic_iv_icu/patients.csv.gz from https://physionet.org/content/mimiciv/3.1/

# 2. Run Processing Script

To process the data run the following command:

```bash
python preprocessing.py
```
# 3. Run Modelling Script

To train models run the following command:
```bash
python train_evaluate.py
```
# 4. Output and Results

The script will generate performance figures: AUROC plots, calibration curves and net benefit plots as well as Shapley values, which will be saved automatically in the figures/ directory.
The repository contains the following key components:

- `preprocessing.py` – Handles data cleaning, feature extraction, and preprocessing of ECG-derived features and demographic data. Performs stratified splitting across age, gender, and target labels, ensuring train/validation/test sets are balanced and representative.

- `train_evaluate.py` – Responsible for model training and evaluation. Trains multilabel classifiers using XGBoost, performs recursive feature elimination for each label, computes performance metrics.

- `stratify.py` – Contains functions for stratified subset sampling and multi-label stratification. Ensures that subsets of the dataset are balanced across combined clinical and demographic labels, supporting consistent and fair training and validation splits.
Note: This file is adapted from the [AI4HealthUOL/CardioDiag](https://github.com/AI4HealthUOL/CardioDiag) repository. 
