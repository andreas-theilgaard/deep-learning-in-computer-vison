from torchmetrics.classification import BinaryAccuracy,MulticlassAccuracy,BinaryAUROC,AUROC,BinaryF1Score,MulticlassF1Score,Precision,BinaryPrecision,BinaryRecall,Recall

class METRICS:
    def __init__(self,config):
        self.config = config
        self.acc = BinaryAccuracy().to(config.device) if config.n_classes in [1,2] else MulticlassAccuracy(num_classes=config.n_classes).to(config.device)
        self.f1 = BinaryF1Score().to(config.device) if config.n_classes in [1,2] else MulticlassF1Score(num_classes=config.n_classes).to(config.device) #micro
        self.precision = BinaryPrecision().to(config.device) if config.n_classes in [1,2] else Precision(task="multiclass",num_classes=config.n_classes).to(config.device) # micro
        self.recall = BinaryRecall().to(config.device) if config.n_classes in [1,2] else Recall(task="multiclass",num_classes=config.n_classes).to(config.device) # micro

    def get_metrics(self, metric_type, preds, targets):
        if hasattr(self, metric_type):
            metric = getattr(self, metric_type)
            return metric(preds, targets)
        else:
            raise ValueError(f"Metric '{metric_type}' not found")