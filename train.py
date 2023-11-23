import torch
import pandas as pd
import os, sys
import argparse
import psutil
import datetime
import pytorch_lightning as pl
from fingerprint_data_module import FingerprintDataModule
from CHEMData import ChemData
from fingerprint_classifier import FingerprintClassifier
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torch import cuda, optim, nn, utils, Tensor

#-----------------------------------------------------------------------------
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(name)-30s] %(message)s ",
    handlers=[logging.StreamHandler()],
    level=logging.INFO)



def predict_with_ids(model, dataset):
    model.eval()
    predictions = []
    compound_ids = []
    smiles_list = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            # Fetch data from dataset
            fingerprint, target = dataset[idx]
            compound_id = dataset.df.iloc[idx][dataset.id_column]  # Fetch compound ID from DataFrame
            smiles = dataset.df.iloc[idx][dataset.smiles_column]   # Fetch SMILES from DataFrame
            
            # Process and predict
            fingerprint = fingerprint.unsqueeze(0).to(device)
            output = model(fingerprint)
            prediction = torch.sigmoid(output)
            
            predictions.append(prediction.item())
            compound_ids.append(compound_id)
            smiles_list.append(smiles)

    return compound_ids, smiles_list, predictions






def main(file_path):
    # Load the data and perform initial processing
    df = ChemData.load_csv(file_path)
    df_processed = ChemData.preprocess_data(df, target_column='target', n_bits=2048)
    ChemData.split_and_save_dataset(file_path, target_column='target', n_bits=2048)
    imbalance_info = ChemData.check_dataset_imbalance(df_processed, 'target')
    
    # Define the model configuration
    config = {
        'fingerprint_size': 2048,
        'dropout': 0.2,
        'learning_rate': 1e-3,
        'max_epochs': 10,
        'batch_size': 5,
    }
    
    # Initialize the logger, data module, and model
    filebase = file_path.rsplit('.', 1)[0] # Remove the file extension
    logger = TensorBoardLogger("tb_logs", name="my_model")
    data_module = FingerprintDataModule(filebase=filebase, target_column='target', batch_size=config['batch_size'])
    model = FingerprintClassifier(config=config, pos_weight=torch.tensor([imbalance_info]))
    
    # Initialize the trainer and fit the model
    trainer = pl.Trainer(max_epochs=config['max_epochs'], logger=logger)
    trainer.fit(model, datamodule=data_module)

    # After training is complete, you can generate predictions
    test_dataset = data_module.test_dataset  # Accessing the test dataset
    compound_ids, smiles_list, predictions = predict_with_ids(model, test_dataset)


    # Optionally, save predictions to a CSV or process further
    predictions_df = pd.DataFrame({
        'COMPOUND_ID': compound_ids,
        'SMILES': smiles_list,
        'PREDICTION': predictions
    })
    predictions_df['BINARY_PREDICTION'] = (predictions_df['PREDICTION'] > 0.5).astype(int)
    predictions_df.to_csv('predictions.csv', index=False)







#==============================================================================
if __name__ == "__main__":
    #-----------------------------------------------------------------------------
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train the model on molecular data.')
    parser.add_argument('file_path', type=str, help='The file path to the CSV data file.')
    args = parser.parse_args()
    main(args.file_path)
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
    endTime = datetime.datetime.now()
    endmem = psutil.Process(os.getpid()).memory_info().rss / 1000000
    logger.info(f"Script ended. Time taken: {endTime - startTime}")
    logger.info(f"Final memory usage: {endmem:.1f} MB")




