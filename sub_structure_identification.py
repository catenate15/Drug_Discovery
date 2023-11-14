import argparse
import datetime
import logging
import os
import pandas as pd
import psutil
import sys
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision('medium')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from fingerprint_data_module import FingerprintDataModule
from fingerprint_classifier import FingerprintClassifier
from CHEMData import ChemData, MolecularDataset


#-----------------------------------------------------------------------------
# Initialize logging
#-----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
 
#------------------------------------------------------- 



def main(filebase, target_column):
#-------------------------------------------------------
    # Configuration
    config = {
        "fingerprint_size": 2048,
        "nclasses": 1,
        "max_epochs": 50,
        "learning_rate": 1e-3,
        "dropout": 0.2,
        "batch_size": 32,
    }

    # Load and preprocess the dataset
    train_df = ChemData.load_csv(f"{filebase}_Train.csv")
    # You might want to check imbalance based on the training set
    is_imbalanced = ChemData.check_dataset_imbalance(train_df, target_column)

    # DataModule Initialization
    data_module = FingerprintDataModule(
        filebase=filebase,
        fingerprint_size=config['fingerprint_size'],
        batch_size=config['batch_size']
    )

    # Calculate the positive weight for imbalanced datasets
    num_positives = train_df[target_column].sum()
    num_negatives = len(train_df) - num_positives
    pos_weight = num_negatives / num_positives if num_positives > 0 else 1

    # Setup TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Model Initialization
    model = FingerprintClassifier(
        fingerprint_size=config['fingerprint_size'],
        dropout_rate=config['dropout'],
        learning_rate=config['learning_rate'],
        use_weighted_loss=is_imbalanced,
        pos_weight=torch.tensor([pos_weight]).to(device) if is_imbalanced else None
    )


    # Initialize Trainer with TensorBoard Logger
    trainer = pl.Trainer(max_epochs=config['max_epochs'], logger=logger)

    # After training
    trainer.fit(model, data_module)

    # Save the model weights
    model_path = "model_weights.pth"
    torch.save(model.state_dict(), model_path)


#==============================================================================
if __name__ == "__main__":
    #-----------------------------------------------------------------------------
    startTime = datetime.datetime.now()
    startmem = psutil.Process(os.getpid()).memory_info().rss / 1000000

    logger.info("Script started.")
    logger.info(f"Initial memory usage: {startmem:.1f} MB")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {cuda.is_available()}")
    if cuda.is_available():
        logger.info(f"CUDA device: {cuda.get_device_name(0)}")

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--filebase", type=str, required=True, help="Base filename for the datasets (without _Train, _Valid, _Test suffix)")
    parser.add_argument("--target_column", type=str, default="target", help="Name of the target column in the dataset.")
    args = parser.parse_args()

    # Run the main function
    main(args.filebase, args.target_column)

    endTime = datetime.datetime.now()
    endmem = psutil.Process(os.getpid()).memory_info().rss / 1000000
    logger.info(f"Script ended. Time taken: {endTime - startTime}")
    logger.info(f"Final memory usage: {endmem:.1f} MB")
