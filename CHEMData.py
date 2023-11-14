import pandas as pd
from torch.utils.data import Dataset
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

class ChemData:
    @staticmethod
    def load_csv(file_path):
        """
        Load CSV file containing compound IDs and SMILES strings.
        """
        df = pd.read_csv(file_path)
        return df
    
    @staticmethod
    def check_dataset_imbalance(df, target_column, imbalance_threshold=0.20):
        """
        Check if the dataset is imbalanced based on the minority class proportion.
        """
        class_counts = df[target_column].value_counts()
        minority_class_proportion = class_counts.min() / class_counts.sum()
        return minority_class_proportion < imbalance_threshold

    @staticmethod
    def smiles_to_morgan(smiles, n_bits=2048):
        """
        Convert SMILES string to Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None]*n_bits  # Return None vector if the molecule is invalid
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))

    @staticmethod
    def preprocess_data(df, target_column='target', imbalance_threshold=0.20):
        """
        Preprocess the data: Generate fingerprints from SMILES, extract targets, and check for class imbalance.
        """
        compound_ids = df['compound_chembl_id']
        smiles = df['canonical_smiles']
        targets = df[target_column]
        fingerprints = df['canonical_smiles'].apply(lambda x: ChemData.smiles_to_morgan(x))

        # Check for class imbalance
        is_imbalanced = ChemData.check_dataset_imbalance(df, target_column, imbalance_threshold)

        return compound_ids, smiles, targets, fingerprints, is_imbalanced

class MolecularDataset(Dataset):
    def __init__(self, compound_ids, fingerprints, targets):
        """
        Create a Dataset from compound IDs, molecular fingerprints, and targets.
        """
        self.compound_ids = compound_ids
        self.fingerprints = torch.tensor(fingerprints.tolist()).float()
        self.targets = torch.tensor(targets.values).float()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.compound_ids[idx], self.fingerprints[idx], self.targets[idx]

# Example Usage:
# df = ChemData.load_csv('path_to_your_csv_file.csv')
# compound_ids, smiles, targets, fingerprints, is_imbalanced = ChemData.preprocess_data(df, 'target_column_name')
