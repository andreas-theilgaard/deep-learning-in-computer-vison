How to run the hotdog classifier models
==============================

We utilize hydra to track the experiments.

For the hotdog classifier we use the `src/configs/HotDogClassifier.yaml` file.

The parameters can easily be overwritten using the command line.
To use W&B set the wandb.use_wandb flag to true.

Examples
------------
```
python src/hotdog_classifier/main.py --config-name='HotDogClassifier.yaml' model='BASIC_CNN' enable_dropout=True dropout_rate=0.25
```
Given around 0.82 test accuracy

The efficientnet model can be used using
```
python src/hotdog_classifier/main.py --config-name='HotDogClassifier.yaml' params.model='efficientnet' params.enable_dropout=True params.dropout_rate=0.25 params.lr=0.0001 params.bs=64 params.epochs=100 params.image_size=128 freeze_params=0.9 pretrained=True
```
Around 0.87

The best model is saved at `outputs/model_name/data/time/.....ckpt`

