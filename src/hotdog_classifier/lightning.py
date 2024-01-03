from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.hotdog_classifier.model_utils import get_model,configure_optimizer,get_loss,get_probs_and_preds
import torch
from src.hotdog_classifier.metrics import METRICS

class HotDogClassifier(pl.LightningModule):
    def __init__(self, config):
        super(HotDogClassifier,self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = get_model(config)
        self.criterion = torch.nn.CrossEntropyLoss() if config.params.n_classes != 1 else torch.nn.BCEWithLogitsLoss()
        self.evaluator = METRICS(config=config)
        self.model_checkpoint = ModelCheckpoint(
            monitor = f"test_{config.track_metric}",
            verbose = config.params.verbose,
            filename = "{epoch}_{val_loss:.4f}",
        )

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        return configure_optimizer(config=self.config,model=self.model)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = get_loss(outputs=outputs,labels=labels,criterion=self.criterion,config=self.config)
        _,preds = get_probs_and_preds(outputs=outputs,config=self.config)
        self.log_metrics(type_='train',loss=loss,preds=preds,labels=labels)
        return loss   
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = get_loss(outputs=outputs,labels=labels,criterion=self.criterion,config=self.config)
        _,preds = get_probs_and_preds(outputs=outputs,config=self.config)
        self.log_metrics(type_='test',loss=loss,preds=preds,labels=labels)
        return loss   
    
    def log_metrics(self,type_,loss,preds,labels):
        self.log(f"{type_}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.config.metrics:
            if metric != 'loss':
                metric_val = self.evaluator.get_metrics(metric_type=metric,preds=preds,targets=labels)
                self.log(f"{type_}_{metric}", metric_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        