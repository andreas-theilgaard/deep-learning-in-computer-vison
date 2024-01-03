import hydra
import logging
from src.hotdog_classifier.utils import create_description
import torch
from tqdm import tqdm
from src.hotdog_classifier.dataloaders import get_data
import pytorch_lightning as pl
from src.hotdog_classifier.lightning import HotDogClassifier
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base="1.2", config_path="../configs", config_name="HotDogClassifier.yaml")
def main(config):
    if config.debug:
        log.setLevel(logging.CRITICAL + 1)

    create_description(config)
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg["runtime"]["output_dir"]
    training_args = config.params

    # model
    model = HotDogClassifier(config)

    # dataloaders
    trainloader,testloader = get_data(training_args)

    WANDB = WandbLogger(
            name=f"{config.params.model}_{config.experiment_name}",
            project='dtu_dlcv',
            config=OmegaConf.to_container(cfg=config,resolve=True,throw_on_missing=True)
        )

    trainer = pl.Trainer(
        devices=config.n_devices, 
        accelerator=config.device, 
        max_epochs = config.params.epochs,
        log_every_n_steps = config.log_every_n,
        callbacks=[model.model_checkpoint],
        logger=WANDB,
    ) 

    trainer.fit(model, trainloader, val_dataloaders=testloader)


if __name__ == "__main__":
    main()
