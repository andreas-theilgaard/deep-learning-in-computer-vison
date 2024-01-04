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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


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

    # dataloaders
    trainloader,testloader,normalizer = get_data(training_args)

    # model
    model = HotDogClassifier(config,normalizer=normalizer)

    WANDB = WandbLogger(
            name=f"{config.params.model}_{config.experiment_name}",
            project='dtu_dlcv',
            config=OmegaConf.to_container(cfg=config,resolve=True,throw_on_missing=True)
        )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=f"test_{config.track_metric}",
        mode="max" if config.track_metric != 'loss' else "min",
        dirpath=save_path,
        filename="{epoch:02d}-{test_acc:.2f}",
    )

    early_stop_callback = EarlyStopping(monitor=f"test_{config.track_metric}", min_delta=0.00, patience=30, verbose=True, mode="max")

    trainer = pl.Trainer(
        devices=config.n_devices, 
        accelerator=config.device, 
        max_epochs = config.params.epochs,
        log_every_n_steps = config.log_every_n,
        callbacks=[checkpoint_callback,early_stop_callback],
        logger=WANDB if config.wandb.use_wandb else None,
        inference_mode=False
    ) 

    trainer.fit(model, trainloader, val_dataloaders=testloader)

    trainer.test(ckpt_path='best',dataloaders=testloader)


if __name__ == "__main__":
    main()
