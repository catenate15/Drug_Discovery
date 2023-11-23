import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryF1Score, BinaryCohenKappa, BinaryMatthewsCorrCoef, BinarySpecificity, BinaryAUROC, BinaryCohenKappa
from pytorch_lightning.loggers import TensorBoardLogger



class FingerprintClassifier(pl.LightningModule):
    def __init__(self, config, pos_weight=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.pos_weight = pos_weight
        self.input_size = self.config['fingerprint_size']
        self.dropout = self.config['dropout']

        # Define the network layers
        self.in_layer = nn.Sequential(
            nn.Linear(self.input_size,512),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.h1_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.h2_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(64, 1)
        )



        # Metrics - for binary classification, num_classes is generally not required
        self.metrics = {
            'accuracy': BinaryAccuracy(),
            'f1_score': BinaryF1Score(),
            'cohen_kappa': BinaryCohenKappa(),
            'mcc': BinaryMatthewsCorrCoef(),
            'specificity': BinarySpecificity(),
            'auroc':  BinaryAUROC()
        }

    def forward(self, x):
        x = self.in_layer(x)
        x = self.h1_layer(x)
        x = self.h2_layer(x)
        return self.out_layer(x)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        # Let's assume you want to decay the learning rate at 50% and 75% of the max epochs
        #milestones = [self.config['max_epochs'] // 2, 3 * self.config['max_epochs'] // 4]
        #lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return [optimizer]

    def criteria(self, Y_pred,Y,pos_weight=None):
        lossFunc = nn. BCEWithLogitsLoss(pos_weight=pos_weight)
        return lossFunc(Y_pred,Y)


    def compute_metrics(self, logits, Y, prefix):
        metric_dict = {f"{prefix}_{name}": metric(logits, Y) for name, metric in self.metrics.items()}
        return metric_dict


    def training_step(self, batch, batch_idx):
        X, Y = batch
        Y_pred = self.forward(X)
        Y = Y.unsqueeze(1)
        loss = self.criteria(Y_pred, Y, pos_weight=self.pos_weight)
        metrics = self.compute_metrics(Y_pred, Y, "train")  # Compute metrics
        logDict = {"train_loss": loss, **metrics}  # Merge loss and metrics into one dictionary
        self.log_dict(logDict, on_step=True, on_epoch=True)  # Log metrics
        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        Y_pred = self.forward(X)
        Y = Y.unsqueeze(1)
        loss = self.criteria(Y_pred, Y, pos_weight=self.pos_weight)
        metrics = self.compute_metrics(Y_pred, Y, "val")  # Compute metrics
        logDict = {"val_loss": loss, **metrics}  # Merge loss and metrics into one dictionary
        self.log_dict(logDict, on_step=False, on_epoch=True)  # Log metrics for each epoch
        return {"val_loss": loss, "log": metrics}  # Return metrics for callbacks or further processing



    def test_step(self, batch, batch_idx):
        X, Y = batch
        Y_pred = self.forward(X)
        Y = Y.unsqueeze(1)
        loss = self.criteria(Y_pred, Y, pos_weight=self.pos_weight)
        metrics = self.compute_metrics(Y_pred, Y, "test")  # Compute metrics for the test set
        self.log_dict({"test_loss": loss, **metrics}, on_step=False, on_epoch=True)  # Log test metrics
        return {"test_loss": loss, "log": metrics}  # Return metrics for callbacks or further processing





