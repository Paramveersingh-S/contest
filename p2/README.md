# Problem 2: Black Box Adversarial Attack on CNN

This folder contains the solution for Problem 2 of the competition.

## Overview

The goal is to generate 1,000 adversarial images against a black-box CNN model that only returns class labels (decision-based / hard-label attack). 

We utilized the **HopSkipJumpAttack** algorithm via the `foolbox` library, which is highly effective in the decision-based setting. The attack estimates the gradient at the decision boundary and takes steps to minimize the $L_2$ distortion, successfully misclassifying the input images while keeping perturbations visually imperceptible.

## File Structure

- `p2_code.ipynb`: The primary Jupyter Notebook containing the attack logic and image generation pipeline.
- `report.pdf`: A short report explaining the methodology and parameter tuning.
- `run_attack.py`: A script used during development to efficiently test and run the attack.
- `data/p2_data/adv_images/`: Directory containing all 1,000 generated adversarial images (named `<class>_<idx>_adv.png`).

## Requirements

To run the notebook or script, you will need the following dependencies installed in your environment:
```bash
pip install torch torchvision numpy pillow foolbox tqdm
```

## Reproduction

1. Make sure `p2_data.zip` has been extracted to `./data/p2_data/`.
2. Open `p2_code.ipynb` and execute all cells. 
3. The attack will utilize your available GPU if `cuda` is detected, otherwise it falls back to `cpu`. 
4. Once completed, exactly 1,000 adversarial images will be saved in `./data/p2_data/adv_images/`.
