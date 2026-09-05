#!/bin/bash
set -e

echo "Running end-to-end HPC Backdoor Detector Pipeline..."

echo "1. Generating EDA Notebook"
python generate_eda.py
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_eda.ipynb
cd ..

echo "2. Training Pipeline (Models, Synth Data, Feature Eng, Thresholding)"
python train_pipeline.py

echo "3. Running smoke test on train set..."
python src/infer.py --csv_path ../trace.csv --output smoke_test_predictions.csv

echo "Pipeline executed successfully!"
