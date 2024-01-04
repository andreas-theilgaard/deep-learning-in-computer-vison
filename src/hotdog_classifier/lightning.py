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
                    ax.imshow(images[i].permute(1,2,0).detach().cpu().numpy())
                    ax.axis('off')
                    ax.set_title(f"Pred: {self.mapper[preds[i].detach().item()]}\nLabel: {self.mapper[labels[i].detach().item()]}")
                plt.tight_layout()   
                #wandb_logger.log_image(key="sample_images", images=images, caption=captions)
                self.logger.experiment.log({f"{type_} Batch {batch_idx} predictions": wandb.Image(fig)})  
                plt.close(fig)
        if type_=='Final_test':
            if batch_idx in [7]:

                for saliency_method in ['VanillaGrad']:

                    if saliency_method == 'VanillaGrad':
                        fig,axes = plt.subplots(5,5,figsize=(16,10))
                        for i, ax in enumerate(axes.flatten()):
                            if i>=25:
                                break
                            saliency = self.VanillaGradient(images[i])
                            ax.imshow(saliency,cmap='hot')
                            ax.axis('off')

                        plt.tight_layout()   
                        #wandb_logger.log_image(key="sample_images", images=images, caption=captions)
                        #self.logger.experiment.log({f"{type_} Batch {batch_idx} predictions": wandb.Image(fig)})  
                        plt.savefig("ged.png")


    def VanillaGradient(self,image,target_class=None):
        import numpy as np
        self.eval()
        with torch.set_grad_enabled(True):
            image = image.unsqueeze(0)
            image_tensor = image.clone()
            image_tensor.requires_grad = True
            image_tensor.retain_grad()

            logits = self.model(image_tensor)
            target = torch.zeros_like(logits)

            target[0][target_class if target_class else logits.topk(1, dim=1)[1]] = 1
            self.model.zero_grad()
            logits.backward(target)
            return image_tensor.grad.detach().cpu()[0].permute(1,2,0).numpy()

    #smoothgrad

    # def saliency_map(self,image,saliency_method='VanillaGrad'):
    #     import numpy as np
    #     self.eval()
    #     image = image.unsqueeze(0)
    #     image = image.requires_grad_(True)#torch.autograd.Variable(image.to(self.config.device), requires_grad=True)

    #     if saliency_method =='VanillaGrad':
    #         output = self(image)
    #         index = torch.argmax(output)
    #         # one_hot = torch.zeros((1, output.size()[-1]))
    #         # one_hot[0][index] = 1
    #         one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    #         one_hot[0][index] = 1
    #         one_hot_tensor = torch.from_numpy(one_hot).to(self.config.device).requires_grad_(True) #torch.nn.Parameter(one_hot.to(self.config.device))#torch.autograd.Variable(one_hot.to(self.config.device), requires_grad=True)
    #         #one_hot = torch.sum(one_hot * output)
    #         one_hot_tensor = (one_hot_tensor * output).sum()
    #         one_hot_tensor.backward()
    #         grad = image.grad.data.cpu().numpy()
    #         grad = grad[0, :, :, :]
    #         return grad

        # n_samples = 25
        # for _ in range(n_samples):
        #     image = torch.autograd.Variable(image,requires_grad=True)
        #     output = self(image)
        #     output_idx = output.argmax(dim=1)
        #     output_max = output[0, output_idx]
        #     image.grad.data.zero_()
        #     output_max.backward(retain_graph=True)
        #     grad = image.grad.data
        #     total_gradients += grad * grad
        # smooth_saliency = total_gradients[0,:,:,:] / n_samples     
        # mask99 = torch.quantile(smooth_saliency, 0.99)
        # smooth_saliency = torch.clamp(smooth_saliency, min=0, max=mask99)
        # smooth_saliency = (smooth_saliency - smooth_saliency.min()) / (smooth_saliency.max() - smooth_saliency.min())
        # return smooth_saliency


    # def saliency_map(self, image):
    #     self.eval()
    #     with torch.set_grad_enabled(True):
    #         image = image.unsqueeze(0)
    #         image.requires_grad = True
    #         output = self(image)
    #         logit, _ = torch.max(output, 1)
    #         logit.backward()
    #         saliency, _ = torch.max(torch.abs(image.grad[0]), dim=0)
    #         saliency = (saliency - saliency.min())/(saliency.max()-saliency.min())

    #         if self.config.params.normalize:
    #             with torch.no_grad():
    #                 image = self.normalizer.denormalize(image)
    #     return image,saliency
