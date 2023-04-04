import argparse
import importlib
import masking,data,utils,models.SCL_model as SCL_model
import os
import torch
import pickle
import wandb
from data import IRAVENDataModule
from models.SCL_model import SCL
importlib.reload(masking)
importlib.reload(data)
importlib.reload(utils)
importlib.reload(SCL_model)
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



parser=argparse.ArgumentParser()
parser.add_argument('--test',help='Run test sweep',action='store_true')
parser.add_argument('--dataset-name',help='name of dataset')


args=parser.parse_args()
print(f'\n\n Running sweep for {args.dataset_name} \n\n ')






#setup
test=args.test
setup_kwargs={
    'model_name':'SCL_90',
    'task_name':args.dataset_name,
    'ckpt_path':'model_ckpts/pretrain_SCL/SCL_pretrain_80.ckpt',
    'save_freq':1 if test else 10,
    'logit_init':2.5,
    'batch_size':8,
    'logging':True,
    'dataset_split':(90,10,0),
    'log_alpha_values':[-10,-5] if test else [-10,-6,-5.66,-5.33,-5],
}




#dataset setup
if 1:
    #task dataset
    task_path=os.path.join('datasets',setup_kwargs.get('task_name'))
    data_module=IRAVENDataModule(batch_size=setup_kwargs.get('batch_size'),split=setup_kwargs.get('dataset_split'))
    data_module.prepare_data()
    data_module.setup(root_dir=task_path)
    train_dataloader_task,test_dataloader_task=data_module.train_dataloader(),data_module.test_dataloader()

    #NOT task dataset
    path_='datasets/originals_masking'
    data_module_=IRAVENDataModule(batch_size=setup_kwargs.get('batch_size'),split=setup_kwargs.get('dataset_split'))
    data_module_.prepare_data()
    data_module_.setup(root_dir=path_)
    test_dataloader_not_task=data_module_.test_dataloader()

#model setup
if 1:
    scl_kwargs={
        "image_size":160,                            # size of image
        "set_size": 9,                               # number of questions + 1 answer
        "conv_channels": [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
        "conv_output_dim": 80,                       # model dimension, the output dimension of the vision net
        "attr_heads": 10,                            # number of attribute heads
        "attr_net_hidden_dims": [128],               # attribute scatter transform MLP hidden dimension(s)
        "rel_heads": 80,                             # number of relationship heads
        "rel_net_hidden_dims": [64,32,5] 
    }
    model=SCL(**scl_kwargs)
    state_dict=utils.get_SCL_state_dict(setup_kwargs.get('ckpt_path'))
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()



def sweep_function(test=test,debug=False):
    
    #use debug to turn logging off
    if not debug:
        logger=wandb.init(project='AVR')


    task_train_dataloader=train_dataloader_task
    task_test_dataloader=test_dataloader_task
    _task_test_dataloader=test_dataloader_not_task

    
    savedir=os.path.join('masks',setup_kwargs.get('model_name'),setup_kwargs.get('task_name'))
    if not os.path.isdir(savedir):
        os.makedirs(savedir)

        

    init_kwargs={
        'model':model,
        'train_dataloader':train_dataloader_task,
        'test_dataloader1':test_dataloader_task,
        'test_dataloader2':test_dataloader_not_task,
        'device':device,
        'savedir':savedir,
        'logit_init':setup_kwargs.get('logit_init'),
    }

    #train kwargs setup
    train_kwargs={
        'alpha':10**-5 if debug else 10**wandb.config.log_alpha,
        'lr':1e-2,
        'tau':1,
        'n_epochs':2 if test else 90,
        'n_batches':10 if test else 'full',
        'val_every_n_steps':2 if test else (len(train_dataloader_task)-1), #val every epoch
        'n_val_batches':2 if test else 'full',
        'eval_every_n_steps':1e10, 
        'n_eval_batches':2 if test else 100,
        'save_freq_epoch':setup_kwargs.get('save_freq'),
        'logging':setup_kwargs.get('logging'),
        'sweep':True,
        'sweep_logger':logger
        }
    
    if wandb.config.log_alpha==setup_kwargs.get('log_alpha_values')[0]:
        
        hyperparams={
            'setup_kwargs':setup_kwargs,
            'train_kwargs':train_kwargs,
        }
        
        fname=os.path.join(savedir,'hyperparameters.pkl')
        with open(fname,'wb') as f:
            pickle.dump(hyperparams,f)
            
        print('Saved sweep hyperparameters')
            
    
    
    masked_scl=masking.MaskedSCLModel(init_kwargs)
    masked_scl.train(**train_kwargs)
    




sweep_name='_'.join([setup_kwargs.get('model_name'),setup_kwargs.get('task_name'),'test' if test else ''])
sweep_configuration={
    'method':'grid',
    'name':sweep_name,
    'metric':{
        'goal':'maximize',
        'name':'Accuracy/validation_task',
        },
    'parameters':{
        'log_alpha':{'values':setup_kwargs.get('log_alpha_values')},
        }
    }



if 1:
    sweep_id=wandb.sweep(sweep=sweep_configuration,project='AVR')
    wandb.agent(sweep_id,function=sweep_function)
    wandb.finish()



if 1:
    sweep_id=wandb.sweep(sweep=sweep_configuration,project='AVR')
    wandb.agent(sweep_id,function=sweep_function)
    wandb.finish()