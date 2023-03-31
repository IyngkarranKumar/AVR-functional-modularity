import argparse
import importlib
import masking,data,utils,models.SCL_model as SCL_model
import os
import torch
import wandb
from data import IRAVENDataModule
from models.SCL_model import SCL
importlib.reload(masking)
importlib.reload(data)
importlib.reload(utils)
importlib.reload(SCL_model)
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



parser=argparse.ArgumentParser()
parser.add_argument('dataset_name',help='name of dataset')


args=parser.parse_args()
print(f'\n\n Running sweep for {args.dataset_name} \n\n ')






#setup
test=True
model_ckpt='/Users/iyngkarrankumar/Documents/AI/AVR-functional-modularity/SCL_pretrain_80.ckpt'
task_path=f'datasets/{args.dataset_name}'
save_freq= 1e12 if test else 1e12
logit_init=0.9
logging=True
batch_size=8

#dataset setup
if 1:
    #task dataset
    path=task_path
    data_module=IRAVENDataModule(batch_size=batch_size)
    data_module.prepare_data()
    data_module.setup(root_dir=path)
    train_dataloader_task,test_dataloader_task=data_module.train_dataloader(),data_module.test_dataloader()

    #NOT task dataset
    path_='datasets/originals_masking'
    data_module_=IRAVENDataModule(batch_size=batch_size)
    data_module_.prepare_data()
    data_module_.setup(root_dir=path_)
    test_dataloader_not_task=data_module_.test_dataloader()

#model setup
if 1:
    model_kwargs={
        "image_size":160,                            # size of image
        "set_size": 9,                               # number of questions + 1 answer
        "conv_channels": [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
        "conv_output_dim": 80,                       # model dimension, the output dimension of the vision net
        "attr_heads": 10,                            # number of attribute heads
        "attr_net_hidden_dims": [128],               # attribute scatter transform MLP hidden dimension(s)
        "rel_heads": 80,                             # number of relationship heads
        "rel_net_hidden_dims": [64,32,5] 
    }
    model=SCL(**model_kwargs)
    state_dict=utils.get_SCL_state_dict(model_ckpt)
    model.load_state_dict(state_dict)
    model.to(device=torch.device('cpu')) #because mps is weird
    model.eval() #for batch norm




def sweep_function(test=test,debug=False):

    if not debug:
        run=wandb.init(project='AVR')

    #for ease of access in WandB UI

    task_train_dataloader=train_dataloader_task
    task_test_dataloader=test_dataloader_task
    _task_test_dataloader=test_dataloader_not_task

    

    task_name=os.path.basename(task_path)
    savedir=os.path.join('masks/SCL',task_name)

    init_kwargs={
    'model':model,
    'train_dataloader':train_dataloader_task,
    'test_dataloader1':test_dataloader_task,
    'test_dataloader2':test_dataloader_not_task,
    'device':device,
    'savedir':'model_ckpts/FFN',
    'logit_init':logit_init,
    }

    #train kwargs setup
    train_kwargs={
        'alpha':1e-5 if debug else wandb.config.alpha,
        'lr':1e-2,
        'tau':1,
        'n_epochs':2 if test else 100,
        'n_batches':5 if test else 'full',
        'val_every_n_steps':2 if test else (len(train_dataloader_task)-1), #val every epoch
        'n_val_batches':2 if test else 'full',
        'eval_every_n_steps':1e10, 
        'n_eval_batches':2 if test else 100,
        'save_freq':save_freq,
        'logging':logging,

        }
    
    print(init_kwargs)
    print(train_kwargs)


        
    masked_scl=masking.MaskedSCLModel(init_kwargs)
    masked_scl.train(**train_kwargs)



alpha_values=[1e-10,1e-5] if test else [1e-10,1e-6,1e-5,1e-4,1e-3,1e-2]
sweep_configuration={
    'method':'grid',
    'name':f'SCL_{args.dataset_name}',
    'metric':{
        'goal':'maximize',
        'name':'validation_accuracy',
        },
    'parameters':{
        'alpha':{'values':alpha_values},
        },
    'description':None
    }


if 1:
    sweep_id=wandb.sweep(sweep=sweep_configuration,project='AVR')
    wandb.agent(sweep_id,function=sweep_function)
    wandb.finish()


