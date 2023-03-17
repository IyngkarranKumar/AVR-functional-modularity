#%%
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import utils
import importlib
import torch.nn.functional as F
import matplotlib as mpl

importlib.reload(utils)



def mask_num_ones(mask):
    num_ones=sum([torch.bincount(torch.flatten(tens.to(torch.int32)),minlength=2)[-1] for tens in mask.values()])
    return num_ones.item()


def get_modal_mask(logit_mask,n=10):
    '''

    Parameters:
        logit_mask: dictionary of logit masks
        n: Number of times to transform logit_mask
    Returns:
        modal_masks: Binary mask - The modal binary mask when logit tensors are transformed n times
    
    '''

    modal_masks={k:torch.zeros_like(v) for k,v in logit_mask.items()}

    mask_list=[utils.transform_logit_tensors(logit_mask) for _ in range(n)]

    for tens_name in logit_mask.keys():
        tens_list=[mask[tens_name] for mask in mask_list] #generate n binary masks from logit mask
        tens_stack=torch.stack(tens_list)
        tens_stack=tens_stack.to(device=torch.device('cpu')) #torch.mode() only works for tensors on cpu
        mode=torch.mode(tens_stack,dim=0)[0]
        modal_masks[tens_name]=mode

    return modal_masks
    

def find_true_weights(model_task_masks):

    print('Ensure that input masks are the modal masks...')

    '''
    Parameters:
        model_task_masks: MODAL masks found for model-task

    Returns:
        shared_weights: the weights shared between the masks
        max_shared_perc: Percentage of shared weights that are also in the mask with fewest weights
    
    '''
    names=list(next(iter(model_task_masks.items()))[-1])
    shared_weights={}

    for idx,tens_name in enumerate(names):
        
        tensor_list=[mask[tens_name].to(torch.int32) for mask in model_task_masks.values()] #get binary tensors for this name

        #find shared
        tens_shared=(torch.ones(size=tensor_list[0].size(),dtype=torch.int32))
        for t in tensor_list:
            tens_shared=torch.logical_and(tens_shared,t)
        tens_shared=tens_shared.int()

        #print([torch.bincount(torch.flatten(tens.to(torch.int32)))[-1] for tens in tensor_list]) #number of ones for each tensor in tens_list
        #print(torch.bincount(torch.flatten(tens_shared.to(torch.int32)))[-1]) #number of ones in tens_shared

        shared_weights[tens_name]=tens_shared.int()


    num_ones_masks=[mask_num_ones(mask) for mask in model_task_masks.values()]
    num_ones_shared=mask_num_ones(shared_weights)
    max_shared_perc=(num_ones_shared/min(num_ones_masks))*100


    true_weights={}

    for mask_name,mask in model_task_masks.items():
        masks_iter=list(zip(mask.values(),shared_weights.values()))
        name_masks_iter = list(zip(names,masks_iter))
        true_weights[mask_name]={k:(v0-v_shared).to(torch.int32) for k,(v0,v_shared) in name_masks_iter}


    return true_weights,shared_weights,max_shared_perc


def sharing_matrix(model_task_masks):

    true_weights,shared_weights,_=find_true_weights(model_task_masks)

    #find true shared weights
    #loop over masks
    true_shared_weights={}
    for idxa,maska_name in enumerate(true_weights):
        for idxb,maskb_name in enumerate(true_weights):
            if idxb<idxa:
                continue
            maska=true_weights[maska_name]
            maskb=true_weights[maskb_name]
            ab_shared_weights={}

            #loop over layers in mask
            layer_names=list(maska.keys())
            for name in layer_names:
                tens_a,tens_b=maska[name],maskb[name]
                tens_shared=(torch.ones(size=(tens_a.size()),dtype=torch.int32))
                for tens in [tens_a,tens_b]:
                    tens_shared=torch.logical_and(tens_shared,tens)
                tens_shared=tens_shared.int()

                ab_shared_weights[name]=tens_shared
            
            true_shared_weights[f'{maska_name},{maskb_name}']=ab_shared_weights

            
    true_shared_names=true_shared_weights.keys()

    max_true_sharing_percentage={}
    for name in true_shared_names:
        maska,maskb=name.split(',')
        norm_const=min([mask_num_ones(true_weights[maska]),mask_num_ones(true_weights[maskb])])
        max_shared_perc_ab = round(((mask_num_ones(true_shared_weights[name]))/norm_const)*100,2)
        max_true_sharing_percentage[name]=max_shared_perc_ab

    n_masks=len(model_task_masks)
    data_arr=np.zeros(shape=(n_masks,n_masks))

    count=0
    for i in range(n_masks):
        n_get=n_masks-i
        subarr=list(max_true_sharing_percentage.values())[count:n_get+count]
        data_arr[i,i:]=subarr
        count+=n_get

    data_arr_low_tril=np.transpose(copy.deepcopy(data_arr))
    np.fill_diagonal(data_arr_low_tril,0)

    plot_matrix=data_arr+data_arr_low_tril

    fig,ax=plt.subplots()
    fig.suptitle('Model maximum shared weight percentages')
    x_labels=y_labels=list(model_task_masks.keys())
    fmt='.1f' #use to get rid of standard form
    cmap='viridis_r'

    plot_kwargs={
        'data':plot_matrix,
        'vmax':100,
        'vmin':0,
        'annot':True,
        'ax':ax,
        'xticklabels':x_labels,
        'yticklabels':y_labels,
        'fmt':fmt,
        'cmap':cmap
    }


    sns.heatmap(**plot_kwargs)



def sparsity(binary_mask):
    
    return utils.sparsity(binary_mask.values())

def localisation(binary_mask):

    def normed_var(mask_tens):
        d=len(mask_tens.size())
        mask_idxs=mask_tens.nonzero().to(torch.float32)
        n=len(mask_idxs)
        if n==0:
            return 0
        mu=torch.mean(mask_idxs,dim=0)

        normed_var=(1/n*d)*sum([(F.mse_loss(idx,mu).item())/(torch.dot(mu,mu).item()+1e-12) for idx in mask_idxs])
        return normed_var
    
    normed_vars=[normed_var(mask_tens) for mask_tens in binary_mask.values()]

    return sum(normed_vars)


def plot_metrics(model_task_masks_iter,plot_type='model',dummy_data=False):

    if plot_type!='model' or plot_type!='task':
        raise Exception(f'Please enter valid arguments for plot_type. Entered values were plot_type:{plot_type}')

    sparsity_metric=[]
    localisation_metric=[]
    model_names=[]
    task_names=[]

    if not dummy_data:
        for model_name,model_task_masks in model_task_masks_iter.items():
            for task_name,task_mask in model_task_masks.items():
                sparsity_metric.append(sparsity(task_mask))
                localisation_metric.append(localisation(task_mask))
                model_names.append(model_name)
                task_names.append(task_name)

    if plot_type=='model':
        mapping={name:i for i,name in enumerate(set(model_names))}
        c=[mapping[model_name] for model_name in model_names]
    elif plot_type=='task':
        mapping={name:i for i,name in enumerate(set(task_names))}
        c=[mapping[task_name] for task_name in task_names]

    plt.scatter(sparsity_metric,localisation_metric,c=c,s=200,alpha=0.75,marker='x')




if 0:
    sizes=[(4,4),(3,1),(2,2),(15,2),(3,1)]
    model_logit_dict={f'layer {idx}':utils.random_binary_tensor(size=sizes[idx]) for idx in range(len(sizes))}
    model_task_masks=utils.generate_model_masks(model_logit_dict)
    true_weights,_,_=find_true_weights(model_task_masks)
    mask0=true_weights['Mask 0']
    sparsity0=sparsity(mask0)
    sharing_matrix(model_task_masks)


if 1: #scl model and maskedmodel setup

    import masking,data,utils
    from data import IRAVENDataModule
    importlib.reload(masking)
    importlib.reload(data)
    importlib.reload(utils)

    from models.SCL_model import SCLTrainingWrapper,SCL

    model_ckpt='/Users/iyngkarrankumar/Documents/AI/AVR-functional-modularity/SCL_pretrain_80.ckpt'
    task_path='datasets/light'
    save_freq= 10000
    batch_size=8
    split=(85,15,0)
    device=torch.device('cpu')

    #dataset setup
    if 1:
        #task dataset
        path=task_path
        data_module=IRAVENDataModule(batch_size=batch_size,split=split)
        data_module.prepare_data()
        data_module.setup(root_dir=path)
        train_dataloader_task,test_dataloader_task=data_module.train_dataloader(),data_module.test_dataloader()
        x,y,*rest=next(iter(train_dataloader_task))

        #NOT task dataset
        path_='datasets/originals_masking'
        data_module_=IRAVENDataModule(batch_size=batch_size,split=split)
        data_module_.prepare_data()
        data_module_.setup(root_dir=path_)
        test_dataloader_not_task=data_module_.test_dataloader()

    #setup model
    if 1:
        scl_kwargs={
            "image_size":160,                            # size of image
            "set_size": 9,                               # number of questions + 1 answer
            "conv_channels": [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
            "conv_output_dim": 80,                       # model dimension, the output dimension of the vision net
            "attr_heads": 10,                            # number of attribute heads
            "attr_net_hidden_dims": [128],               # attribute scatter transform MLP hidden dimension(s)
            "rel_heads": 80,                             # number of relationship heads
            "rel_net_hidden_dims": [64, 32, 5] 
        }
        model=SCL(**scl_kwargs)

        #load
        state_dict=utils.get_SCL_state_dict(model_ckpt)
        model.load_state_dict(state_dict)
        model.eval() #for batch norm

    #initialise masked model
    if 1: 
        init_kwargs={
            'model':model,
            'train_dataloader':train_dataloader_task,
            'test_dataloader1':test_dataloader_task,
            'test_dataloader2':test_dataloader_not_task,
            'device':device,
            'savedir':'model_ckpts/FFN',
        }

        masked_scl=masking.MaskedSCLModel(init_kwargs)


if 0:
    sizes=[(4,4),(3,1),(2,2),(15,2),(3,1)]
    model_logit_dict={f'layer {idx}':utils.random_binary_tensor(size=sizes[idx]) for idx in range(len(sizes))}
    model_task_masks=utils.generate_model_masks(model_logit_dict)
    mask0=model_task_masks['Mask 0']

    mask_tens=list(mask0.values())[0].int()
    d=len(mask_tens.size())
    mask_idxs=mask_tens.nonzero().to(torch.float32)
    n=len(mask_idxs)
    mu=torch.mean(mask_idxs,dim=0)

    normed_var=(1/n*d)*sum([(F.mse_loss(idx,mu).item())/torch.dot(mu,mu).item() for idx in mask_idxs])

#%%

modelmodel_logit_dict={f'layer {idx}':utils.random_binary_tensor(size=sizes[idx]) for idx in range(len(sizes))}
test_task_masks=utils.generate_model_masks(model_logit_dict)
scl_task_masks=utils.generate_model_masks(masked_scl)
model_task_masks_iter={'test model':model_task_masks,
                        'scl': scl_task_masks}


sparsity_metric=[]
localisation_metric=[]
model_names=[]
task_names=[]



#calculate metrics
if 1:

    for model_name,model_task_masks in model_task_masks_iter.items():
        for task_name,task_mask in model_task_masks.items():
            print(model_name,task_name)
            sparsity_metric.append(sparsity(task_mask))
            localisation_metric.append(localisation(task_mask))
            model_names.append(model_name)
            task_names.append(task_name)

#%%
plot_type='model'
if plot_type=='model':
    mapping={name:i for i,name in enumerate(set(model_names))}
    c=[mapping[model_name] for model_name in model_names]
elif plot_type=='task':
    mapping={name:i for i,name in enumerate(set(task_names))}
    c=[mapping[task_name] for task_name in task_names]


plt.style.use('ggplot')
labelsize=20
titlesize=20
ticksize=14
cmap=plt.cm.jet
cmaplist=[cmap(i) for i in range(cmap.N)]
cmap=mpl.colors.LinearSegmentedColormap.from_list(
    'Custom cmap', cmaplist, cmap.N)
cb=plt.colorbar.ColorBase(ax,cmap=cmap)

fontdict1={'size':labelsize}
fontdict2={'size':ticksize}

fig,ax=plt.subplots()
ax.scatter(sparsity_metric,localisation_metric,c=c,s=200,alpha=0.75,marker='x')
ax.set_xlabel('Sparsity',fontdict=fontdict1)
ax.set_ylabel('Localisation',fontdict=fontdict1)
ax.set_xticklabels(labels=ax.get_xticklabels(),fontdict=fontdict2)
ax.set_yticklabels(labels=ax.get_yticklabels(),fontdict=fontdict2)
ax.set_title('Sparsity vs Localisation',fontsize=titlesize)
ax.legend()
fig.tight_layout()

