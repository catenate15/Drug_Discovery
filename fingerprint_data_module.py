import pytorch_lightning as pl
from torch.utils.data import DataLoader
from CHEMData import ChemData, MolecularDataset

class FingerprintDataModule(pl.LightningDataModule):
    def __init__(self, filebase, fingerprint_size=2048,  target_column='target', batch_size=32, num_workers=0):
        super().__init__()
        self.filebase = filebase
        self.fingerprint_size = fingerprint_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_column = target_column


    def setup(self, stage=None):
        # Assuming that the dataset files are named following a specific pattern
        self.train_dataset = MolecularDataset(f'{self.filebase}_Train.csv', self.fingerprint_size, target_column= self.target_column)
        self.val_dataset = MolecularDataset(f'{self.filebase}_Valid.csv', self.fingerprint_size, target_column=self.target_column)
        self.test_dataset = MolecularDataset(f'{self.filebase}_Test.csv', self.fingerprint_size, target_column=self.target_column)
        print("Dataset loading is completed")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

