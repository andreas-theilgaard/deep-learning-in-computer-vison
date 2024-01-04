from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.hotdog_classifier.model_utils import get_model,configure_optimizer,get_loss,get_probs_and_preds
import torch
from src.hotdog_classifier.metrics import METRICS
import matplotlib.pyplot as plt
import wandb
import colorcet as cc
import numpy as np


class HotDogClassifier(pl.LightningModule):
    def __init__(self, config,normalizer=None):
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
        self.mapper = {0:'Hotdog',1:'Not Hotdog'}
        self.normalizer=normalizer

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        optimizer = configure_optimizer(config=self.config,model=self.model)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = get_loss(outputs=outputs,labels=labels,criterion=self.criterion,config=self.config)
        _,preds = get_probs_and_preds(outputs=outputs,config=self.config)
        self.log_metrics(type_='train',loss=loss,preds=preds,labels=labels,images=images,batch_idx=batch_idx)
        return loss   
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = get_loss(outputs=outputs,labels=labels,criterion=self.criterion,config=self.config)
        _,preds = get_probs_and_preds(outputs=outputs,config=self.config)
        self.log_metrics(type_='test',loss=loss,preds=preds,labels=labels,images=images,batch_idx=batch_idx)
        return loss
    
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)
        images, labels = batch
        outputs = self(images)
        loss = get_loss(outputs=outputs,labels=labels,criterion=self.criterion,config=self.config)
        _,preds = get_probs_and_preds(outputs=outputs,config=self.config)
        self.log_metrics(type_='Final_test',loss=loss,preds=preds,labels=labels,images=images,batch_idx=batch_idx)
        return loss     
    
    def log_metrics(self,type_,loss,preds,labels,images=None,batch_idx=None):
        self.log(f"{type_}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        for metric in self.config.metrics:
            if metric != 'loss':
                metric_val = self.evaluator.get_metrics(metric_type=metric,preds=preds,targets=labels)
                self.log(f"{type_}_{metric}", metric_val, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.config.wandb.use_wandb:
            if (self.trainer.current_epoch % 20 == 0 and batch_idx in [4,16]) or (type_=='Final_test' and batch_idx in [0,2,4,6,8,10,16,20]):
                fig,axes = plt.subplots(5,5,figsize=(16,10))
                for i, ax in enumerate(axes.flatten()):
                    if i>=25:
                        break
                    image_normalized = images[i] if not self.config.params.normalize else self.normalizer.denormalize(images[i])
                    ax.imshow(image_normalized.permute(1,2,0).detach().cpu().numpy())
                    ax.axis('off')
                    pred,label = self.mapper[preds[i].detach().item()],self.mapper[labels[i].detach().item()]
                    color = 'green' if pred==label else 'red'
                    ax.set_title(f"Pred: {pred}\nLabel: {label}",color=color)
                plt.tight_layout()   
                #wandb_logger.log_image(key="sample_images", images=images, caption=captions)
                self.logger.experiment.log({f"{type_} Batch {batch_idx} predictions": wandb.Image(fig)})  
                plt.close(fig)
            if type_=='Final_test':
                if batch_idx in [0,2,4,6,8,10,16,20]:
                    fig = self.plot_grid(images=images[0:4,:])
                    self.logger.experiment.log({f"SmoothGrad {type_} Batch {batch_idx}": wandb.Image(fig)})  

    def get_mask(self,image_tensor, target_class=None):
        image_tensor = image_tensor.clone()
        image_tensor.requires_grad = True
        image_tensor.retain_grad()

        logits = self.model(image_tensor)
        target = torch.zeros_like(logits)
        target[0][target_class if target_class else logits.topk(1, dim=1)[1]] = 1
        self.model.zero_grad()
        logits.backward(target)
        return image_tensor.grad.detach().cpu()[0].permute(1,2,0).numpy()

    def SmoothGrad(self,image,std,n_samples,target_class=None):
        if len(image.shape)!=4:
            image = image.unsqueeze(0)
        image_tensor = image.clone()
        std = std * (torch.max(image_tensor) - torch.min(image_tensor)).detach().cpu().numpy()
        batch, channels, width, height = image_tensor.size()
        grad_sum = torch.zeros((width, height, channels))
        for sample in range(n_samples):
            noise = torch.empty(image_tensor.size()).normal_(0, std).to(image_tensor.device)
            noise_image = image_tensor + noise
            grad_sum += self.get_mask(image_tensor=noise_image, target_class=target_class)
        saliency_map = grad_sum / n_samples
        return saliency_map

    def normalize(self,mask, vmin=None, vmax=None, percentile=99):
        if vmax is None:
            vmax = np.percentile(mask, percentile)
        if vmin is None:
            vmin = np.min(mask)
        return (mask - vmin) / (vmax - vmin + 1e-10)

    def show_mask(self,mask, title='', cmap=None, alpha=None, norm=True, axis=None):
        if norm:
            mask = self.normalize(mask)
        (vmin, vmax) = (-1, 1) if cmap == cc.cm.bkr else (0, 1)
        axis.imshow(mask, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, interpolation='lanczos')
        if title:
            axis.set_title(title)
        axis.axis('off')

    def make_grayscale(self,mask):
        return np.sum(mask, axis=2)

    def make_black_white(self,mask):
        return self.make_grayscale(np.abs(mask))

    def plot_grid(self,images,std_list=[0,0.05,0.1,0.2,0.3,0.5],n_samples=25,norm=True):
        std_ranges = ['image']+std_list
        fig,axes = plt.subplots(images.shape[0],len(std_ranges),figsize=(15,8),tight_layout=True)
        for image_i in range(images.shape[0]):
            for col_index,std in enumerate(std_ranges):
                if std != "image":
                    saliency_map = self.SmoothGrad(image=images[image_i],std=std,n_samples=n_samples)
                    bw_mask = self.make_black_white(saliency_map.numpy())
                    if norm:
                        bw_mask = self.normalize(bw_mask)
                        (vmin, vmax) = (0, 1)
                    axes[image_i,col_index].imshow(bw_mask, cmap=cc.cm.gray, alpha=None, vmin=vmin, vmax=vmax, interpolation='lanczos')
                else:
                    image_normalized = images[image_i] if not self.config.params.normalize else self.normalizer.denormalize(images[image_i])
                    axes[image_i,col_index].imshow(image_normalized.cpu().permute(1,2,0).numpy())

                if image_i==0:
                    axes[image_i,col_index].set_title(std)
                axes[image_i,col_index].axis('off')
        return fig

