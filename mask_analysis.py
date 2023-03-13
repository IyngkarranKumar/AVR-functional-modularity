import torch
import numpy
import torchvision
import matplotlib.pyplot
import copy
import utils
import importlib

importlib.reload(utils)


paths=[]

def get_modal_mask(logit_mask,n=10):

    modal_masks={k:torch.zeros_like(v) for k,v in logit_mask.items()}

    mask_list=[utils.transform_logit_tensors(logit_mask) for _ in range(n)]

    for tens_name in logit_mask.keys():
        tens_list=[mask[tens_name] for mask in mask_list] #generate n binary masks from logit mask
        tens_stack=torch.stack(tens_list)
        tens_stack=tens_stack.to(device=torch.device('cpu')) #torch.mode() only works for tensors on cpu
        mode=torch.mode(tens_stack,dim=0)[0]
        modal_masks[tens_name]=mode

    return modal_masks
    

def identify_shared_weights():
    pass

def sparsity():
    pass

def localisation():
    pass

def share_matrix():
    pass