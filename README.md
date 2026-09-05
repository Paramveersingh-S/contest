# IIT Kharagpur Hackathon 2026 🚀

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

Welcome to our submission for the **IIT Kharagpur Hackathon 2026**. 
This repository contains our implementations for the online phase problems.

## 🎯 Problem 1: Detection of Backdoor Attacks using Hardware Performance Counters (HPCs)

Our flagship solution addresses the unsupervised anomaly detection of backdoor attacks based on microarchitectural data-flow dynamics using HPCs.

### Architecture
We employ a robust ensemble methodology combining multiple model families to capture different aspects of anomaly representation:

- **Autoencoder**: For complex nonlinear structure and multi-feature co-shifts.
- **Gaussian Mixture Model (GMM)**: For global distributional shift evaluation.
- **Isolation Forest**: To identify sparse/rare local outliers.
- **One-Class SVM**: For establishing tight nonlinear boundaries around the normal manifold.
- **Local Outlier Factor (LOF)**: For localized density anomalies.

These scores are normalized and passed through a meta-ensemble (e.g., Logistic Stacker) for high-confidence predictions, calibrated against a principled **Synthetic Anomaly Validation Set** constructed due to the absence of backdoored traces during training.

### 🏛 Repository Structure
- `p1/hpc-backdoor-detector/`: Contains all code for Problem 1.
  - `src/`: Data cleaning, feature engineering, modeling, and evaluation code.
  - `notebooks/`: Exploratory Data Analysis (EDA) and experimental notebooks.
  - `models/`: Serialized pre-trained artifacts.
  - `data/`: Datasets (CSV files).
  - `report/`: The final solution report.

### 🚀 Instructions for Judges

To test and run the solution for Problem 1, you can use our easy-to-use runner script:

```bash
cd p1/hpc-backdoor-detector
pip install -r requirements.txt
./run_pipeline.sh
```

To run inference on a new trace file, you can utilize the `infer.py` entry point:
```bash
python src/infer.py --csv_path <path_to_new_trace_csv>
```

---

## 🔒 Problem 2: Black-Box Adversarial Attack

This directory handles the adversarial attack track.
- `p2/`: Contains artifacts and notebooks for Problem 2.

*More details to be added as this track is implemented.*
