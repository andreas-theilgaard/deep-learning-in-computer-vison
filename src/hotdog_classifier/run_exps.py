import subprocess
BASE = "python src/hotdog_classifier/main.py --config-name='HotDogClassifier.yaml' wandb.tag='Experiments'"

# Consider data aug on offf for all


# BASIC_CNN
for augment_type in ['No','Yes','Extra']:
    if augment_type == 'No':
        augment = False
        extra_augment = False
    elif augment_type == 'Yes':
        augment = True
        extra_augment = False      
    elif augment_type == 'Extra':
        augment = True
        extra_augment = True   
    for apply_batchnorm in [True,False]:
        for apply_dropout in [True,False]:
            exp_name = f"BASIC_CNN_enable_dropout={apply_dropout}_apply_batchnorm={apply_batchnorm}_augment={augment}_extra_augment={extra_augment}"
            subprocess.call(f"{BASE} models='BASIC_CNN' experiment_name={exp_name} models.training.enable_dropout={apply_dropout} models.training.apply_batchnorm={apply_batchnorm} data_params.augment={augment} data_params.augment={extra_augment}",
                    shell=True,
                )
# efficientnet
# for freeze_params in [0.7,0.9,1.0]:
#     for apply_dropout in [True,False]:
#         exp_name = f"efficientnet_enable_dropout={apply_dropout}freeze_params={freeze_params}"
#         subprocess.call(f"{BASE} models='BASIC_CNN' experiment_name={exp_name} models.training.enable_dropout={apply_dropout} models.training.freeze_params={freeze_params}",
#                 shell=True,
#             )     

#googlenet
# subprocess.call(f"{BASE} models='googlenet' experiment_name='googlenet'",
#                 shell=True,
#             )    