# CardioCXR
This repository contains scripts and resources for preprocessing and modeling ECG data to perform predictive tasks related to cardiopulmonary health from combined ECG and chest X-ray (CXR) data.

## Overview

This project integrates ECG features with chest X-ray labels to build predictive multilabel classification models using XGBoost. The repository provides tools to:

- Clean and preprocess ECG-realted features and CXR data  
- Engineer relevant ECG features and synchronize them with CXR metadata  
- Stratify datasets by clinical variables (age, gender, diagnosis labels)  
- Train and evaluate multilabel classifiers with performance metrics and uncertainty estimates  
- Interpret model outputs via SHAP values for explainability  

---
This repository contains scripts to process and model ECG data for predictive tasks. Follow the steps below to replicate the experiments.
# 1. Download Required Datasets

Before running the scripts, download the following files and place them in the root directory of this repository:

mimic-cxr-2.0.0-metadata.csv.gz from https://physionet.org/content/mimic-cxr/2.0.0/

cxr-lt-2024/labels.csv from https://physionet.org/content/cxr-lt-iccv-workshop-cvamd/2.0.0/

machine_measurements.csv from https://physionet.org/content/mimic-iv-ecg/1.0/

mimic_iv_icu/patients.csv.gz from https://physionet.org/content/mimiciv/3.1/

# 2. Run Processing Script

To process the data run the following command:

python preprocessing.py

# 3. Run Modelling Script

To train models run the following command:

python train_evaluate.py

# 4. Output and Results

The script will generate performance figures: AURCOC plots, calibration curves and net benefit plots as well as Shapley values, which will be saved automatically in the figures/ directory.
The repository contains the following key components:

- `preprocessing.py`  
  Scripts for data cleaning, feature extraction, and preprocessing of ECG-derived features and demographic data as well as stratified spliting across age, gender and target labels and train val test split.

- `train_evaluate.py`  
  Scripts to train multilabel classifiers using XGBoost, perform recursive feature elimination for each label, evaluate model performance (AUROC with bootstrapped confidence intervals), and generate interpretation plots using SHAP.
  
- `stratify.py`  
  Contains functions for stratified subset sampling and multi-label stratification. It is used in the preprocessing pipeline to ensure balanced and representative splits based on combined clinical and demographic labels.
Note: This file is adapted from the AI4HealthUOL/CardioDiag repository. 
