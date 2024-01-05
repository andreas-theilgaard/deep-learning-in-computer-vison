import torch
from src.hotdog_classifier.models import BASIC_CNN,Pretrained

def configure_optimizer(config,model):
    if config.params.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr = config.params.lr,weight_decay=config.params.weight_decay)
    elif config.params.optimizer == 'SGD':
        return torch.optim.SGD(model.parameters(),lr=config.params.lr,weight_decay=config.params.weight_decay)    

def get_model(config):
    if config.params.model == 'BASIC_CNN':
        return BASIC_CNN(config.params)
    elif config.params.model in ['efficientnet','googlenet']:
        return Pretrained(config=config)

def get_loss(outputs,labels,criterion,config):
    if config.params.n_classes ==1:
        loss = criterion(outputs.squeeze(1),labels.float())
    else:
        loss = criterion(outputs,labels)
    return loss

def get_probs_and_preds(outputs,config):
    if config.params.n_classes ==1:
        probs = torch.sigmoid(outputs)
        preds = preds = (probs>=0.5).float()
    else:
        probs,preds = torch.max(torch.softmax(outputs.data,dim=-1),dim=-1)
    return probs,preds
 

# def train(trainloader,model,optimizer,criterion,fabric,config):
#     model.train()

#     y_hat_list = []
#     y_true_list = []
#     loss_list = []
#     probs_list = []

#     for batch in trainloader:
#         inputs, labels = batch

#         # Zero gradients
#         optimizer.zero_grad()

#         # forward pass in network
#         outputs = model(inputs)
#         loss = get_loss(outputs=outputs,labels=labels,criterion=criterion,config=config)
#         fabric.backward(loss)
#         optimizer.step()

#         probs,preds = get_probs_and_preds(outputs,config)

#         y_hat_list.append(preds)
#         y_true_list.append(labels)
#         probs_list.append(probs)
#         loss_list.append(loss)  

#     y_hat = torch.cat(y_hat_list).to(config.device)
#     y_true = torch.cat(y_true_list).to(config.device)
#     probs = torch.cat(probs_list).to(config.device)
#     avg_loss = torch.tensor(loss_list).mean()

#     predictions = {'y_true':y_true,'y_hat':y_hat,'prob':probs}
#     return avg_loss,predictions

# @torch.no_grad()
# def infer(loader,model,criterion,config):
#     model.eval()

#     y_hat_list = []
#     y_true_list = []
#     loss_list = []
#     probs_list = []
#     for batch in loader:
#         inputs, labels = batch

#         # forward pass in network
#         outputs = model(inputs)
#         loss = get_loss(outputs=outputs,labels=labels,criterion=criterion,config=config)
#         probs,preds = get_probs_and_preds(outputs,config)

#         y_hat_list.append(preds)
#         y_true_list.append(labels)
#         probs_list.append(probs)
#         loss_list.append(loss)  

#     y_hat = torch.cat(y_hat_list).to(config.device)
#     y_true = torch.cat(y_true_list).to(config.device)
#     probs = torch.cat(probs_list).to(config.device)
#     avg_loss = torch.tensor(loss_list).mean()

#     predictions = {'y_true':y_true,'y_hat':y_hat,'prob':probs}
#     return avg_loss,predictions