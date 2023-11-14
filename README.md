# Drug_Discovery
## Antibiotic_drug_discovery
### Molecular Activity Prediction and Activation Mapping
This repository contains a collection of scripts for training a neural network to predict the activity of molecular compounds and for performing activation mapping to understand the contributions of different substructures to the predicted activity.

Overview
The project is split into several modules, each with a specific role:

**CHEMData**: Module for loading and preprocessing chemical data.

**fingerprint_data_module**: PyTorch Lightning data module for handling data loading and batching.

**fingerprint_classifier**: PyTorch Lightning module defining the neural network model for molecular activity prediction.

**main.py**: Main script that orchestrates the training process and saves the model weights.

**activation_map.py**: (Yet to be included) Script for computing activation maps to visualize the contributions of molecular substructures.

### Setup
To use these scripts, you will need Python 3.8+ and the following packages:

pandas
torch
pytorch_lightning
rdkit
torchmetrics
You can install these packages using pip:


**pip install pandas torch pytorch_lightning rdkit-pypi torchmetrics**
##### Usage
To train the model and predict molecular activity:

Ensure your molecular data is in a CSV format with columns for compound IDs, SMILES strings, and activity labels.
Run the main.py script, specifying the base name of your dataset files:

