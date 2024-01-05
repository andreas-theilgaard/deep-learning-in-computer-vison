import torch
from src.hotdog_classifier.models import BASIC_CNN,Pretrained

def configure_optimizer(config,model):
    if config.models.training.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr = config.models.training.lr,weight_decay=config.models.training.weight_decay)
    elif config.models.training.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(),lr=config.models.training.lr,weight_decay=config.models.training.weight_decay)    

def get_model(config):
    if config.models.name == 'BASIC_CNN':
        return BASIC_CNN(config)
    elif config.models.name  in ['efficientnet','googlenet']:
        return Pretrained(config)

def get_loss(outputs,labels,criterion,config):
    if config.n_classes ==1:
        loss = criterion(outputs.squeeze(1),labels.float())
    else:
        loss = criterion(outputs,labels)
    return loss

def get_probs_and_preds(outputs,config):
    if config.n_classes ==1:
        probs = torch.sigmoid(outputs)
        preds = preds = (probs>=0.5).float()
    else:
        probs,preds = torch.max(torch.softmax(outputs.data,dim=-1),dim=-1)
    return probs,preds