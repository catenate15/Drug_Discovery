import pytorch_lightning as pl
from fingerprint_data_module import FingerprintDataModule
from CHEMData import ChemData, MolecularDataset
from fingerprint_classifier import FingerprintClassifier  # Ensure this import is correct

def main():
    # Configuration for the model
    config = {
        'input_size': 2048,  # Assuming this is your fingerprint size
        'nclasses': 2,  # Number of classes
        'dropout': 0.2,  # Dropout rate
        'learning_rate': 1e-3  # Learning rate
    }

    # Initialize the data module
    data = FingerprintDataModule(filebase='PA', fingerprint_size=2048, target_column='target', batch_size=32, num_workers=0)
    data.setup()

    # Initialize the model with the config dictionary
    model = FingerprintClassifier(config=config)

    # Create a trainer and test the model
    trainer = pl.Trainer(max_epochs=1)
    trainer.test(model, datamodule=data)

if __name__ == "__main__":
    main()


