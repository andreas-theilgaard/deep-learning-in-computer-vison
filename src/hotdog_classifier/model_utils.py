import torch
from src.hotdog_classifier.models import BASIC_CNN

def configure_optimizer(config,model):
    if config.params.optimizer == 'Adam':
        return torch.optim.Adam(model.parameters(), lr = config.params.lr,weight_decay=config.params.weight_decay)

def get_model(config):
    if config.params.model == 'BASIC_CNN':
        return BASIC_CNN(config.params.n_classes)  

def train(trainloader,model,optimizer,criterion,fabric,config):
    model.train()

    y_hat_list = []
    y_true_list = []
    loss_list = []
    probs_list = []

    for batch in trainloader:
        inputs, labels = batch

        # Zero gradients
        optimizer.zero_grad()

        # forward pass in network
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        fabric.backward(loss)
        optimizer.step()

        if config.params.n_classes != 1:
            probs,preds = torch.max(torch.softmax(outputs.data,dim=-1),dim=-1)
        else:
            probs,preds = torch.max(torch.sigmoid(outputs.data),dim=-1)

        y_hat_list.append(preds)
        y_true_list.append(labels)
        probs_list.append(probs)
        loss_list.append(loss)  

    y_hat = torch.cat(y_hat_list).to(config.device)
    y_true = torch.cat(y_true_list).to(config.device)
    probs = torch.cat(probs_list).to(config.device)
    avg_loss = torch.tensor(loss_list).mean()

    predictions = {'y_true':y_true,'y_hat':y_hat,'prob':probs}
    return avg_loss,predictions

@torch.no_grad()
def infer(loader,model,criterion,config):
    model.eval()

    y_hat_list = []
    y_true_list = []
    loss_list = []
    probs_list = []
    for batch in loader:
        inputs, labels = batch

        # forward pass in network
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if config.params.n_classes != 1:
            probs,preds = torch.max(torch.softmax(outputs.data,dim=-1),dim=-1)
        else:
            probs,preds = torch.max(torch.sigmoid(outputs.data),dim=-1)

        y_hat_list.append(preds)
        y_true_list.append(labels)
        probs_list.append(probs)
        loss_list.append(loss)  

    y_hat = torch.cat(y_hat_list).to(config.device)
    y_true = torch.cat(y_true_list).to(config.device)
    probs = torch.cat(probs_list).to(config.device)
    avg_loss = torch.tensor(loss_list).mean()

    predictions = {'y_true':y_true,'y_hat':y_hat,'prob':probs}
    return avg_loss,predictions