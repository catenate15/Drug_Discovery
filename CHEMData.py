import pandas as pd
from torch.utils.data import Dataset
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

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
    def split_and_save_dataset(file_path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, target_column='target'):
        df = pd.read_csv(file_path)
        # Splitting the dataset
        train, valid_test = train_test_split(df, train_size=train_ratio)
        valid, test = train_test_split(valid_test, train_size=valid_ratio / (valid_ratio + test_ratio))

        # Constructing new file names by removing the '.csv' extension
        base_path = file_path.rsplit('.', 1)[0]

        # Save the splits
        train.to_csv(f'{base_path}_Train.csv', index=False)
        valid.to_csv(f'{base_path}_Valid.csv', index=False)
        test.to_csv(f'{base_path}_Test.csv', index=False)


        # Constructing new file names by removing the '.csv' extension
        base_path = file_path.rsplit('.', 1)[0]

        # Save the splits
        train.to_csv(f'{base_path}_Train.csv', index=False)
        valid.to_csv(f'{base_path}_Valid.csv', index=False)
        test.to_csv(f'{base_path}_Test.csv', index=False)

class MolecularDataset(Dataset):
    def __init__(self, filename, fingerprint_size, target_column='target'):
        """
        Create a Dataset from a CSV file without pre-processing all fingerprints.
        """
        self.df = pd.read_csv(filename)
        self.fingerprint_size = fingerprint_size
        self.target_column = target_column

    def smiles_to_morgan(self, smiles):
        """
        Convert SMILES string to Morgan fingerprint.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(self.fingerprint_size)  # Return zero vector if the molecule is invalid
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=self.fingerprint_size)
        return torch.tensor(fingerprint, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        compound_id = self.df.iloc[idx]['compound_chembl_id']
        smiles = self.df.iloc[idx]['canonical_smiles']
        fingerprint = self.smiles_to_morgan(smiles)
        target = torch.tensor(self.df.iloc[idx][self.target_column], dtype=torch.float32)
        return compound_id, fingerprint, target



if __name__ == "__main__":
    ChemData.split_and_save_dataset('PA.csv')



