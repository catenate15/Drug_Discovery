# Drug_Discovery
## Antibiotic_drug_discovery
### Molecular Activity Prediction and Activation Mapping
This repository contains a collection of scripts for training a neural network to predict the activity of molecular compounds and for performing activation mapping to understand the contributions of different substructures to the predicted activity.

## Prerequisites

- Python 3.x
- PyTorch
- PyTorch Lightning
- RDKit
- TorchMetrics
- sklearn

## Installation

Install the required libraries using pip:

```sh
pip install torch pytorch-lightning rdkit-pypi torchmetrics scikit-learn
```

## Repository Structure

- `CHEMData.py`: Contains the `ChemData` class for loading, preprocessing, and splitting the dataset.
- `MolecularDataset.py`: Defines a PyTorch `Dataset` for lazy loading and on-the-fly processing of chemical data.
- `FingerprintDataModule.py`: A PyTorch Lightning `DataModule` for setting up the data loaders for the model.
- `FingerprintClassifier.py`: A PyTorch Lightning `Module` defining the deep learning model architecture and training steps.
- `main.py`: The main script that integrates all components, performs model training, and saves the trained model.
- `activation_map.py`: (Yet to be included) Script for computing activation maps to visualize the contributions of molecular substructures.
## Usage

### Step 1: Preprocess and Split the Dataset

Before training the model, you need to split your dataset into training, validation, and test sets. Run the `CHEMData.py` script as follows:

```sh
python CHEMData.py
```

This will create three CSV files: `PA_Train.csv`, `PA_Valid.csv`, and `PA_Test.csv`.

### Step 2: Train the Model

Once the data is split, you can train the model using `main.py`. You will need to specify the base filename of your datasets and the name of the target column.

```sh
python main.py --filebase PA --target_column <target_column_name>
```

Replace `<target_column_name>` with the actual name of the target column in your dataset.

### Step 3: Evaluate the Model

After training, the model's performance can be evaluated using the validation and test datasets through the PyTorch Lightning Trainer methods.

