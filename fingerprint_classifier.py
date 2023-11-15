import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import BinaryAccuracy, BinaryF1Score, CohenKappa, MatthewsCorrcoef, Specificity, AUROC
from pytorch_lightning.loggers import TensorBoardLogger



class FingerprintClassifier(pl.LightningModule):
    def __init__(self, config, pos_weight=None):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.pos_weight = pos_weight

        # Define the network layers
        self.in_layer = nn.Sequential(
            nn.Linear(self.config['fingerprint_size'], 512),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
        self.h1_layer = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
        self.h2_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.config['dropout'])
        )
        self.out_layer = nn.Sequential(
            nn.Linear(64, self.config['nclasses'])
        )

        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.config['pos_weight'])


        # Metrics - for binary classification, num_classes is generally not required
        self.metrics = {
            'accuracy': BinaryAccuracy(),
            'f1_score': BinaryF1Score(),
            'cohen_kappa': CohenKappa(),
            'mcc': MatthewsCorrcoef(),
            'specificity': Specificity(),
            'auroc': AUROC()
        }

    def forward(self, x):
        x = self.in_layer(x)
        x = self.h1_layer(x)
        x = self.h2_layer(x)
        return self.out_layer(x)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['learning_rate'])
        # Let's assume you want to decay the learning rate at 50% and 75% of the max epochs
        milestones = [self.config['max_epochs'] // 2, 3 * self.config['max_epochs'] // 4]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        return [optimizer], [lr_scheduler]


    def compute_metrics(self, logits, Y, prefix):
        metric_dict = {f"{prefix}_{name}": metric(logits, Y) for name, metric in self.metrics.items()}
        return metric_dict

    def shared_step(self, batch, prefix):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_fn(logits, Y)

        metric_dict = self.compute_metrics(logits, Y, prefix)
        metric_dict[f"{prefix}_loss"] = loss

        self.log_dict(metric_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, 'test')