import torch
class metrics:
    def __init__(self,eps:float=1e-8):
        self.eps = eps

    def get_confusion(self,y_hat,mask):
        # assuming y_hat is logits, then convert to confidences using sigmoid
        if y_hat.min().item() < 0.0 or (y_hat.max().item() > 1.0):
            y_hat = torch.sigmoid(y_hat)
        y_hat = (y_hat > 0.50).float()

        self.TP = (y_hat.flatten() * mask.flatten()).sum()
        self.FN = mask[y_hat == 0].sum()
        self.FP = y_hat[mask == 0].sum()
        self.TN = y_hat.numel() - self.TP - self.FN - self.FP

    def get_metrics(self,y_hat,mask):
        self.get_confusion(y_hat,mask)
        dice = ((2 * self.TP) / (2 * self.TP + self.FN + self.FP + self.eps)).item()
        iou = ((self.TP) / (self.TP + self.FN + self.FP )).item()
        acc = (self.TP+self.TN)/(self.TP+self.TN+self.FP+self.FN)
        sensitivity = self.TP/(self.TP+self.FN)
        specificity = self.TN/(self.TN+self.FP)
        self.metric_dict = {'dice':dice,'iou':iou,'acc':acc,'sensitivity':sensitivity,'specificity':specificity}
        return self.metric_dict 
    
    def print_my_metrics(self,y_hat,mask,type_):
        metric_dict = self.get_metrics(y_hat,mask)
        for key in metric_dict:
            print(f"{type_} {key}: {metric_dict[key]}")


class loss_func:
    def __init__(self,config):
        self.type_ = config.loss
        self.gamma = 2
        self.pos_weights = torch.tensor([1/(config.pos_total/config.neg_total)]).to(config.device)
        self.eps = 1e-8
        if self.type_ == 'WeightedBCE':
            print(f"Using pos_weights: {self.pos_weights}")

    def BCE(self):
        return torch.nn.BCEWithLogitsLoss()

    def FocalLoss(self,y_hat,mask):
        y_hat = torch.sigmoid(y_hat)
        return - torch.mean((1-y_hat)**self.gamma  * mask * torch.log(y_hat) + (1-mask)*torch.log(1-y_hat))

    def WeightedBCE(self):
        return torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
    
    def Dice_BCE(self,y_hat,mask):
        BCE = self.BCE()(y_hat,mask)
        y_hat = torch.sigmoid(y_hat)
        y_hat = y_hat.view(-1)
        mask = mask.view(-1)
        intersection = (y_hat * mask).sum()
        DICE_LOSS = 1 - (2.*intersection + self.eps)/(y_hat.sum() + mask.sum() + self.eps)
        Dice_BCE = BCE + DICE_LOSS
        return Dice_BCE
    
    def get_loss(self,y_hat,mask):
        if self.type_ == 'BCE':
            return self.BCE()(y_hat,mask)
        elif self.type_ == 'FocalLoss':
            return self.FocalLoss(y_hat,mask)
        elif self.type_ == 'WeightedBCE':
            return self.WeightedBCE()(y_hat,mask)
        elif self.type_ == 'DICE_BCE':
            return self.Dice_BCE(y_hat,mask)
    
    def loss_name(self):
        return self.type_