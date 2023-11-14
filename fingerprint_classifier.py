import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics import BinaryAccuracy, BinaryF1Score, CohenKappa, MatthewsCorrcoef, Specificity, AUROC
from pytorch_lightning.loggers import TensorBoardLogger



class FingerprintClassifier(pl.LightningModule):
    def __init__(self, fingerprint_size=2048, num_classes=2, dropout_rate=0.2, learning_rate=1e-3, pos_weight=None):
        super().__init__()
        self.save_hyperparameters()

        # Define the network layers
        self.network = nn.Sequential(
            nn.Linear(fingerprint_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Metrics
        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.f1_score = BinaryF1Score()
        self.cohen_kappa = CohenKappa(num_classes=num_classes)
        self.mcc = MatthewsCorrcoef(num_classes=num_classes)
        self.specificity = Specificity(num_classes=num_classes)
        self.auroc = AUROC(num_classes=num_classes)

    def forward(self, x):
        return self.network(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, Y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)        
        self.train_accuracy(logits, Y)
        self.log('train_acc', self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.cohen_kappa(logits, Y)
        self.log('train_kappa', self.cohen_kappa, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.mcc(logits, Y)
        self.log('train_mcc', self.mcc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, Y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.val_accuracy(logits, Y)
        self.log('val_acc', self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, Y)
        self.f1_score(logits, Y)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', self.f1_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
