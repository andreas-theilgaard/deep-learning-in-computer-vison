from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import matplotlib.pyplot as plt
from src.segmentation.models import get_model
from src.segmentation.utils import loss_func,metrics
import numpy as np
import wandb
from IPython.display import clear_output
import matplotlib.colors


class Segmenter(pl.LightningModule):
    def __init__(self, config):
        super(Segmenter,self).__init__()
        self.config = config
        self.save_hyperparameters()
        self.model = get_model(config).to(config.device)
        self.criterion = loss_func(config)
        self.evaluator = metrics()
        self.model_checkpoint = ModelCheckpoint(
            monitor = f"Val_dice",
            verbose =True,
            filename = "{epoch}_{val_loss:.4f}",
        )

    def forward(self,x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = self.config.lr)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion.get_loss(outputs,labels)
        self.log_metrics(type_='Train',loss=loss,preds=outputs,masks=labels,images=images,batch_idx=batch_idx)
        return loss   
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion.get_loss(outputs,labels)
        self.log_metrics(type_='Val',loss=loss,preds=outputs,masks=labels,images=images,batch_idx=batch_idx)

        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion.get_loss(outputs,labels)
        self.log_metrics(type_='Test',loss=loss,preds=outputs,masks=labels,images=images,batch_idx=batch_idx)
        return loss
    

    ################## Log functions ######################
    def log_metrics(self,type_,loss,preds,masks,images=None,batch_idx=None):

        thres = 0.5
        # bincmap = matplotlib.colors.ListedColormap(['blue','blue','blue','red','red','red'])
        
        self.log(f"{type_}_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        metric_dict = self.evaluator.get_metrics(preds,masks)
        for metric in metric_dict:
            self.log(f"{type_}_{metric}", metric_dict[metric], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if type_ in ['Val','Test']:
            if (batch_idx in [0]):
                Y_hat = torch.sigmoid(preds).detach().cpu()
                X = images.detach().cpu()
                Y = masks.detach().cpu()
                
                fig, ax = plt.subplots(nrows=4, ncols=self.config.batch_size, figsize=((self.config.batch_size), (3+1)),dpi=400)
                for k in range(self.config.batch_size):
                    im = np.rollaxis(X[k].numpy(), 0, 3)
                    cmap = 'jet'
                    ax[0,k].imshow(im)
                    ax[1,k].imshow(Y[k, 0], cmap=cmap)
                    ax[2,k].imshow(Y_hat[k, 0], cmap=cmap,vmin=0, vmax=1)

                    maskim = im.copy()
                    maskim[Y_hat[k][0]<=thres] = [0,0,0]
                    
                    implt = ax[3,k].imshow(maskim, cmap=cmap,vmin=0, vmax=1)

                titles = ['Real','Mask','Output','Threshold']
                for i in range(4): 
                    ax[i,0].text(-0.1, 0.5, titles[i],rotation=90,ha='center',va='center',transform=ax[i,0].transAxes)
                for a in ax.flat: 
                    a.set_axis_off()
            
                if type_ == 'Val':
                    plt.suptitle(f"Validation")
                    metricstxt = f'Epoch {self.current_epoch+1} / {self.config.epochs}    loss: {loss:.2f}'
                elif type_ == 'Test':
                    plt.suptitle(f"Test on best validation")
                    metricstxt = f'Total epochs: {self.config.epochs}        loss: {loss:.2f}'
                ax[0,0].text(0, 1.25, metricstxt,ha='left',va='center',transform=ax[0,0].transAxes)
        
                #cbar = fig.colorbar(implt, ax=ax.ravel().tolist(), shrink=1)
                #cbar.set_ticks([0,thres,1])
                #cbar.set_ticklabels(['0',str(thres),'1'])
                #fig.tight_layout()

                if self.config.use_wandb:
                    self.logger.experiment.log({f"{type_} Batch {batch_idx} predictions": wandb.Image(fig)}) 
                else:
                    if (self.current_epoch+1 != self.config.epochs):
                        clear_output(wait=True)
                    plt.show()
                plt.close(fig)

            
        