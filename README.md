# IIT Kharagpur Asian HOST 2026 - Problem Statements

This repository contains the solution for both **Problem 1 (HPC Backdoor Detector)** and **Problem 2 (Black Box Adversarial Attack on CNN)**.

## Problem 1: HPC Backdoor Detection
**Objective:** Detect backdoor attacks in Neural Networks using Hardware Performance Counters (HPCs).

### Methodology
We formulated this as an unsupervised anomaly detection task on the 800 clean traces provided. A robust **Meta-Ensemble** of 5 diverse models (Autoencoder, GMM, Isolation Forest, OCSVM, LOF) was used to reliably classify anomalous behavior without any labeled backdoor data. We synthesized validation anomalies (feature-subset shift, multiplicative noise) to calibrate thresholds.

### Results
The ensemble effectively synthesizes global, local, and density-based anomaly signals.

| Model | AUROC Score |
| :--- | :--- |
| Autoencoder | 0.8334 |
| GMM | 0.8114 |
| Isolation Forest | 0.8007 |
| OCSVM | 0.8052 |
| LOF | 0.8355 |
| **Meta-Ensemble** | **0.8300** |

![Problem 1 ROC Curve](results/roc_curve.png)
*(Caption: Receiver Operating Characteristic (ROC) curve for the Meta-Ensemble evaluating synthetic backend anomalies.)*

---

## Problem 2: Black Box Adversarial Attack on CNN
**Objective:** Perform a decision-based black-box adversarial attack against a CNN to generate 1,000 adversarial images with minimal visual distortion.

### Methodology
We utilized the **HopSkipJumpAttack** (a decision-based attack) to estimate the decision boundary gradient and minimize the $L_2$ distance. The algorithm iteratively moves towards the original image while maintaining misclassification. By capping the steps to 20, we heavily optimized query efficiency (runtime of ~1328s) while successfully maintaining human-imperceptible noise. 

### Results

| Metric | Value |
| :--- | :--- |
| **Attack Success Rate** | **99.60%** |
| **Average $L_2$ Distortion** | **1.0218** |

![Problem 2 Clean vs Adversarial Images](results/clean_vs_adv.png)
*(Caption: Comparison of Clean images (left) vs generated Adversarial images (right) showing virtually indistinguishable $L_2$ distortion of 1.0218.)*

---

**Note**: To reproduce the results, please see `p1/hpc-backdoor-detector/README.md` and `p2/README.md`.
