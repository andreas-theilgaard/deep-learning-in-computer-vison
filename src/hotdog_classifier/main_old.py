# import hydra
# import logging
# from src.hotdog_classifier.utils import LoggerClass
# import torch
# from tqdm import tqdm
# from lightning_fabric import Fabric
# from src.hotdog_classifier.dataloaders import get_data
# from src.hotdog_classifier.model_utils import get_model,configure_optimizer,train,infer

# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)

# @hydra.main(version_base="1.2", config_path="../configs", config_name="HotDogClassifier.yaml")
# def main(config):
#     if config.debug:
#         log.setLevel(logging.CRITICAL + 1)

#     logger = LoggerClass(log=log,cfg=config)

#     logger.create_description(yaml_configuration=config)

#     hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
#     save_path = hydra_cfg["runtime"]["output_dir"]
#     training_args = config.params

#     fabric = Fabric(accelerator=config.device)#,precision="bf16-mixed")
#     fabric.launch()

#     # dataloaders here
#     trainloader,testloader = get_data(training_args)
#     # dataloaders end
#     trainloader = fabric.setup_dataloaders(trainloader)
#     testloader = fabric.setup_dataloaders(testloader)    
#     #fabric.seed_everything(logger.seeds[42])
#     #set_seed(logger.seeds[exp_run])

#     # get model
#     model = get_model(config).to(config.device)
#     optimizer = configure_optimizer(config=config,model=model)
#     # setup model and optimizer
#     criterion = torch.nn.CrossEntropyLoss() if config.params.n_classes != 1 else torch.nn.BCEWithLogitsLoss()
#     model,optimizer = fabric.setup(model,optimizer)
#     model.train()

    
#     data = tqdm(range(training_args.epochs),desc='Model Training')
#     logger.start_run()

#     # do training and inference loop
#     for epoch in data:
#         train_loss, train_predictions = train(trainloader=trainloader,model=model,optimizer=optimizer,criterion=criterion,fabric=fabric,config=config)
#         test_loss,test_predictions = infer(loader=testloader,model=model,criterion=criterion,config=config)

#         predictions = {'train':train_predictions,'test':test_predictions}
#         losses = {'train':train_loss,'test':test_loss}
#         postfix_prog_bar = logger.add_to_run(predictions=predictions,losses=losses)
#         data.set_postfix(postfix_prog_bar)
#     logger.end_run()
#     logger.save_results(save_path + "/results.json")

# if __name__ == "__main__":
#     main()
