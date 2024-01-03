
def accuracy(y_hat,y_true):
    return (y_hat==y_true).float().mean()

# TO DO Implement more metrics
# F1, Precision, Recall, AUC

class Evaluator:
    def __init__(self, metrics_list: list):
        self.metrics_list = metrics_list

    def collect_metrics(self, predictions: dict,losses:dict):
        """
        predictions : {'train':{'logits':.....,'y_true':......}}
        """
        results = {}
        for data_type in predictions.keys():
            inner_results = {}
            y_hat = predictions[data_type]["y_hat"]
            y_true = predictions[data_type]["y_true"]
            for metric in self.metrics_list:
                if metric == "acc":
                    inner_results[metric] = accuracy(y_true=y_true, y_hat=y_hat)
                #elif metric == "f1":
                    #inner_results[metric] = f1(y_true=y_true, y_hat=y_hat)
                # and so on....
            inner_results['loss'] = losses[data_type]
            results[data_type] = inner_results
        return results
    
        