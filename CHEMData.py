import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import ast
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import cuda, optim, nn, utils, Tensor

from sklearn.model_selection import train_test_split

class ChemData:
    @staticmethod
    def load_csv(file_path):
        df = pd.read_csv(file_path)
        return df
    
    @staticmethod
    def check_dataset_imbalance(df, target_column):
        class_counts = df[target_column].value_counts()
        minority_class_proportion = class_counts.min() / class_counts.sum()
        return minority_class_proportion

    @staticmethod
    def smiles_to_morgan(smiles, n_bits=2048):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits))
    
    @staticmethod
    def preprocess_data(df, target_column='target', n_bits=2048):
        # Processing fingerprints and handling invalid SMILES
        df['fingerprint'] = df['canonical_smiles'].apply(lambda x: ChemData.smiles_to_morgan(x, n_bits))
        # Filter out rows where fingerprint is None
        valid_data = df.dropna(subset=['fingerprint'])
        invalid_data = df[df['fingerprint'].isnull()]
        print("Invalid SMILES strings (excluded from dataset):")
        print(invalid_data[['compound_chembl_id', 'canonical_smiles']])
        valid_data['target'] = valid_data[target_column].astype(float)
        return valid_data
    
    @staticmethod
    def split_and_save_dataset(file_path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15, target_column='target', n_bits=2048):
        df = pd.read_csv(file_path)
        df = ChemData.preprocess_data(df, target_column, n_bits)

        # Splitting the dataset with stratification
        stratify_col = df[target_column]
        train, valid_test = train_test_split(df, train_size=train_ratio, stratify=stratify_col)
        # Update stratify for the second split to match the valid_test subset
        valid_test_stratify = stratify_col[valid_test.index]
        valid, test = train_test_split(valid_test, train_size=valid_ratio / (valid_ratio + test_ratio), stratify=valid_test_stratify)

        base_path = file_path.rsplit('.', 1)[0]
        train.to_csv(f'{base_path}_Train.csv', index=False)
        valid.to_csv(f'{base_path}_Valid.csv', index=False)
        test.to_csv(f'{base_path}_Test.csv', index=False)

    # Print class proportions in each set for verification
        for dataset, name in [(train, "Train"), (valid, "Validation"), (test, "Test")]:
            class_counts = dataset[target_column].value_counts(normalize=True)
            print(f"Class proportions in {name} set:")
            print(class_counts)

class MolecularDataset(Dataset):

    def __init__(self, filename, target_column='target', id_column = 'compound_chembl_id', smiles_column = 'canonical_smiles'):
        self.filename = filename
        self.target_column = target_column
        self.id_column = id_column
        self.smiles_column = smiles_column
 
        self.load_mb()


 
    def load_mb(self):
    # --------------------------------------------------------------------------------------    
        self.df = pd.read_csv(self.filename)

        # Convert the fingerprint string back into a list of ints

        self.fp = self.df['fingerprint'].apply(ast.literal_eval).to_list()
        self.target = self.df[self.target_column]
        self.ID = self.df[self.id_column]

        self.X = Tensor(np.array(self.fp,dtype=np.float32))
        self.Y = Tensor(np.array(self.target,dtype=np.float32))


    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



