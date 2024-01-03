import torch
import random
import numpy as np
from watermark import watermark
from datetime import datetime
import wandb
from omegaconf import OmegaConf
import pandas as pd
from metrics import Evaluator
import time

def setup_sweep(sweep_configuration):
    configuration = OmegaConf.load(sweep_configuration)
    configuration = OmegaConf.to_container(configuration)
    sweep_id = wandb.sweep(sweep=configuration,entity="dtu_dlcv")
    return sweep_id

def merge_configs(wandb_config,config):
    wandb_search_values = dict(wandb_config)
    for key in wandb_search_values:
        config.params.training[key] = wandb_search_values[key]
    
def find_best_device():
    if torch.cuda.is_available():
        device = torch.device("gpu")
    elif torch.backends.mps.is_available():
     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def prepare_metric_cols(cfg):
    metric_cols = []
    directions = []
    for metric in cfg.metrics:
        for col in cfg.track_splits:
            metric_cols.append(f"{col.lower()} {metric.lower()}")
    return metric_cols

def get_seeds(n=10):
    return list(range(n))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WB_Logger:
    def __init__(self,cfg,group=None,project=None):
        self.cfg = cfg
        self.group = group
        self.project = project

    def init_WB(self,seed=None):
        wandb.init(
            project=self.project,
            reinit=False,
            group=self.group,
            job_type=f"{seed}",
            anonymous='must',
            save_code=True,
            settings=wandb.Settings(start_method="thread"),
            config=OmegaConf.to_container(cfg=self.cfg,resolve=True,throw_on_missing=True)
        )

    def log_metrics(self,metrics):
        for x in metrics:
            if x != 'Run':
                data_type, metric = x.lower().split(" ")
                wandb.log({f"{data_type} {metric}:": metrics[x]})

class LoggerClass:
    def __init__(self,log=None,cfg=None):
        """
        log: Python logging object (optional)
        cfg: configuration file (optional)
        """
        self.log = log
        self.cfg = cfg

        self.results = pd.DataFrame()
        self.metrics = cfg.metrics
        self.track_metric = cfg.track_metric
        self.runs = cfg.runs
        self.current_run = 0
        self.seeds = get_seeds(self.runs)
        self.log = log

        self.evaluator = Evaluator(self.metrics)
        self.wb_logger = WB_Logger(cfg=self.cfg,group=cfg.wandb.group,project=cfg.wandb.project)

        self.prepared_columns= prepare_metric_cols(self.cfg)


        self.header = lambda x: f"""{'='*x}"""

    def create_description(self,yaml_configuration=None,extra_description=None):
        watermark_description = watermark(author="",
                    watermark=False,
                    conda=False,
                    machine=True,
                    gpu=True,
                    current_time=True,
                    updated=False,
                    python=True,                
                    iversions=False
                    )
        description = '\n' + self.header(50) + '\n'
        description+= f"""Script Executed, {(datetime.now()).strftime("%b-%d %H:%M")}:""" + '\n'*2
        description+= watermark_description

        if yaml_configuration:
            description+='\n'*2
            description+= f"Running experiment on {self.cfg.device}"
            description+= (f"\nConfigurations for current experiment:\n\nConfiguration: \n {OmegaConf.to_yaml(self.cfg)}")
            description+='\n'

        if extra_description:
            description+='\n'*2
            description+=extra_description + '\n'
        description+=self.header(50)

        if self.log:
            self.log.info(description)
        else:
            print(description)

    def log_to_wb(self,key,value):
        wandb.log({key:value})
    
    def start_run(self):
        if self.cfg.wandb.use_wandb and not self.cfg.debug:
            self.wb_logger.init_WB(seed=self.seeds[self.current_run-1])
        self.start = time.time()
        self.current_run += 1
        self.log.info(f"Run {self.current_run}/{self.runs} using seed {self.seeds[self.current_run-1]}")
        self.X = pd.DataFrame(columns=self.prepared_columns + ["Run"])

    def add_to_run(self, predictions,losses):
        results = self.evaluator.collect_metrics(predictions=predictions,losses=losses)
        results_to_add = {}
        results_to_add["Run"] = self.current_run
        for metric in self.metrics:
            for data_type in self.cfg.track_splits:
                results_to_add[f"{data_type.lower()} {metric}"] = float((results[data_type.lower()][metric]).cpu())
        self.X.loc[len(self.X), :] = results_to_add

        if self.cfg.wandb.use_wandb and not self.cfg.debug:
            self.wb_logger.log_metrics(metrics=results_to_add)
        
        update_prog_bar = {f"{x} {self.cfg.track_metric}:":float(results[f"{x.lower()}"][self.cfg.track_metric].cpu()) for x in self.cfg.track_splits}
        update_prog_bar['Train Loss:'] =  float(results['train']['loss'].cpu())
        return update_prog_bar

    def end_run(self):
        end =  time.time()
        self.results = pd.concat([self.results, pd.DataFrame(self.X)])
        elapsed = end - self.start
        if self.cfg.wandb.use_wandb and not self.cfg.debug:
            wandb.log({f"Elapsed Time:": elapsed})

    def save_results(self, save_path):
        self.results.columns = self.prepared_columns + ["Run"]
        self.results.reset_index(drop=True, inplace=True)
        self.results.to_json(save_path)
        if self.cfg.wandb.use_wandb and not self.cfg.debug:
            results_table = wandb.Table(dataframe=self.results)
            wandb.log({"Results": results_table})
            wandb.finish()
