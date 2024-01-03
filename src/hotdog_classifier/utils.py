import torch
import random
import numpy as np
from watermark import watermark
from datetime import datetime
from omegaconf import OmegaConf

def create_description(yaml_configuration=None,extra_description=None):
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
    header = lambda x: f"""{'='*x}"""
    description = '\n' + header(50) + '\n'
    description+= f"""Script Executed, {(datetime.now()).strftime("%b-%d %H:%M")}:""" + '\n'*2
    description+= watermark_description

    if yaml_configuration:
        description+='\n'*2
        description+= f"Running experiment on {yaml_configuration.device}"
        description+= (f"\nConfigurations for current experiment:\n\nConfiguration: \n {OmegaConf.to_yaml(yaml_configuration)}")
        description+='\n'

    if extra_description:
        description+='\n'*2
        description+=extra_description + '\n'
    description+=header(50)


