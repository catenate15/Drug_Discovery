
---

# Antibiotic Drug Discovery Project

## Introduction
This project is dedicated to advancing antibiotic drug discovery by leveraging machine learning techniques to analyze molecular compounds. It aims to address the growing challenge of antimicrobial resistance by identifying promising antibiotic candidates through a computational approach.

## Installation

Before running the project, ensure you have Python 3 and the necessary libraries installed.

1. **Clone the repository:**
   ```
   git clone [repository-url]
   ```
2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Modules Overview

### `CHEMData.py`

This module is responsible for processing chemical data. Key functionalities include loading data from CSV files, converting SMILES strings to Morgan fingerprints, preprocessing datasets, and splitting data into training, validation, and test sets.

### `fingerprint_data_module.py`

Integrates with PyTorch Lightning to manage datasets for training, validation, and testing. It loads data using `MolecularDataset` from `CHEMData.py`, preparing it for use in machine learning models.

### `fingerprint_classifier.py`

Defines a PyTorch Lightning module for a fingerprint-based classifier. It includes the neural network architecture, training, validation, and testing steps, as well as performance metrics.

### `train.py`

The main script for training the model. It initializes and configures the model, data module, and training process. It also includes functionality for logging and monitoring system resource usage.

### `activation_map.py`

Generates activation maps from the trained model to visualize which parts of the molecular fingerprints most influence the model's predictions. This module runs separately from the main training script and outputs activation maps as CSV files.

## Usage

To train the model and generate predictions:

1. **Prepare your dataset** in CSV format with the required columns (e.g., SMILES strings, compound IDs, target labels).

2. **Run the training script:**
   ```
   python train.py [path-to-your-dataset.csv]
   ```
   This will train the model and save the trained weights.

3. **Generate activation maps:**
   ```
   python activation_map.py [path-to-your-dataset.csv]
   ```
   This will produce a CSV file containing activation maps for your dataset.

## Documentation

Each module contains inline comments explaining the purpose of functions and classes. For detailed understanding, refer to these comments within each script.

---




