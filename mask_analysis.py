import torch
import numpy
import torchvision
import matplotlib.pyplot
import copy
import utils
import importlib

importlib.reload(utils)


paths=[]



def mask_num_ones(mask):
    num_ones=sum([torch.bincount(torch.flatten(tens.to(torch.int32)))[0] for tens in mask.values()])
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
    

def identify_shared_weights(model_task_masks):
    
    names=list(next(iter(model_task_masks.items()))[-1].keys())
    shared_weights={}

    for tens_name in names:
        tensor_list=[mask[tens_name].to(torch.int32) for mask in model_task_masks.values()] #get binary tensors for this name

        #find shared
        tens_shared=(torch.ones(size=tensor_list[0].size(),dtype=torch.int32))
        for t in tensor_list:
            tens_shared=torch.logical_and(tens_shared,t)
        tens_shared=tens_shared.int()

        print([torch.bincount(torch.flatten(tens.to(torch.int32)))[-1] for tens in tensor_list]) #number of ones for each tensor in tens_list
        print(torch.bincount(torch.flatten(tens_shared.to(torch.int32)))[-1]) #number of ones in tens_shared



        shared_weights[tens_name]=tens_shared.int()
    
    num_ones_masks=[mask_num_ones(mask) for mask in model_task_masks.values()]
    num_ones_shared=mask_num_ones(shared_weights)
    max_shared_frac=num_ones_shared/min(num_ones_masks)

    print(num_ones_shared,num_ones_masks)

    return shared_weights,max_shared_frac,tensor_list,tens_shared



def sparsity(logit_mask):
    
    modal_binary_mask = (get_modal_mask(logit_mask)).values()
    modal_binary_mask=[b.to(torch.float32) for b in modal_binary_mask]

    return utils.sparsity(modal_binary_mask)

def localisation():
    pass

def share_matrix():
    pass