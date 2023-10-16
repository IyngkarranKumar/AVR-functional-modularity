#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn.functional as F
import random
import io
import pickle


#%% 

def view_matrices(source_dir,type="train",n_view=1):
    subdirs = os.listdir(os.path.join(source_dir,type))
    for subdir in subdirs:
        path = os.path.join(source_dir,type,subdir)
        names = glob.glob(path+"/*.npz")
        files = names[:n_view]
        for img_file in files:
            data = np.load(img_file)
            target = data['target']
            images = data['image']; img_shape = images.shape[1:]

            basename = os.path.split(os.path.basename(img_file))[-1]
            name = subdir+"/"+basename

            gridspec_kw={'hspace':0}
            fig,axs = plt.subplots(4,6,figsize = (6,4))
            matrix_axs = [(0,0),(0,1),(0,2),
                            (1,0),(1,1),(1,2),
                            (2,0),(2,1),(2,2),
                            (0,4),(0,5),
                            (1,4),(1,5),
                            (2,4),(2,5),
                            (3,4),(3,5)]
            fig.subplots_adjust(top=0.9,bottom=0.1,hspace=0,wspace=0.1)
            

            #prepare axes
            for i in range(len(axs)):
                for j in range(len(axs[i])):
                    ax = axs[i,j]
                    ax.tick_params(left=False,right=False,labelleft=False,
                        labelbottom=False,bottom=False)
                    if not [(x,y) for x,y in matrix_axs if x==i and y==j]:
                        ax.axis('off')

            #create matrix
            for img_idx in range(len(images)+1):
                img_ax_idx = matrix_axs[img_idx]; ax = axs[img_ax_idx]
                
                if img_idx==8:
                    img = np.zeros(img_shape)
                    ax.imshow(img,cmap='binary')
                    continue
                elif img_idx < 8:
                    img = images[img_idx]
                elif img_idx > 8:
                    img = images[img_idx-1]

                ax.imshow(img,cmap='gray')

            #fig.tight_layout()

            #fig.suptitle(name,fontsize=15)
            

def imshow(tens):
    arr=tens.permute(1,2,0).numpy()
    plt.imshow(arr)

def calculate_accuracy(pred_logits,targets):
    probs=F.softmax(pred_logits,dim=1)
    pred=torch.argmax(probs,axis=1)
    acc=(len(torch.nonzero(pred==targets))/len(pred))*100
    return acc

def sparsity(mask,logits=False):
    if logits:
        binary_mask=transform_logit_tensors(mask)
        binary_mask_iter=binary_mask.values()
    else:
        binary_mask_iter=mask.values()
    total=sum(torch.numel(b) for b in binary_mask_iter)
    ones=sum(torch.count_nonzero(b) for b in binary_mask_iter)
    sparsity=(total-ones)/total
    return sparsity.item()*100

def mask_num_ones(binaries):

    if type(binaries)==dict:
        binary_mask_iter=binaries.values()
        ones=sum(torch.count_nonzero(b) for b in binary_mask_iter)
    if type(binaries)==torch.Tensor:
        ones=torch.count_nonzero(binaries)


    return ones.item()


def get_named_children(model):

    '''
    IMPORTANT: We assume that a leaf child is one that has 0 children
    This needs checking
    '''
    
    children_dict={}
    named_modules=dict(model.named_modules())
    for module_name,module in named_modules.items():
        if len(list(module.children()))==0:
            children_dict[module_name]=module

    return children_dict


def transform_logit_tensors(logit_dict,tau=1):

    with torch.no_grad():

        samples={}
        for k,v in logit_dict.items():
            U1=torch.rand_like(v,requires_grad=True)
            U2=torch.rand_like(v,requires_grad=True)
            samples[k]=torch.sigmoid((v - torch.log(torch.log(U1) / torch.log(U2))) / tau)
            

        binaries_stop={}
        for k,v in samples.items():
            with torch.no_grad():
                binaries_stop[k]=(v>0.5).float()-v
        
        binaries={}
        for k,v in binaries_stop.items():
            binaries[k]=v+samples[k]

    return binaries


def random_binary_tensor(size):
    t=torch.rand(size)
    t[t<0.5]=0.0
    t[t>=0.5]=1.0
    return t


def generate_random_mask(sizes,type='binary',fill_value=0.9):

    weight_mask={}

    for i,sz in enumerate(sizes):
        if type=='binary':
            t=torch.rand(sz)
            t[t<0.5]=0
            t[t>=0.5]=1
            t=t.int()
        elif type=='logit':
            t=torch.full(size=sz,fill_value=fill_value)
        else:
            raise Exception("Invalid mask type entered.")
        weight_mask[f'layer {i}']=t

    return weight_mask

def generate_multiple_masks(n_masks=10):

    masks=[]
    for _ in range(n_masks):
        sizes=[]
        n_tensors=random.randint(4,10)
        for tens in range(n_tensors):
            tens_sz=(random.randint(1,10),random.randint(1,10))
            sizes.append(tens_sz)
        masks.append(generate_random_mask(sizes))

    return masks
        
def generate_model_masks(input,n_masks=5):

    if isinstance(input,dict):
        logit_mask=input
    else:
        logit_mask=input.logit_tensors_dict
    masks={}
    for _ in range(n_masks):
        masks[f'Mask {_}']={k:random_binary_tensor(v.size()) for k,v in logit_mask.items()}

    return masks

def mask_numel(mask):
    return sum([torch.numel(tens) for tens in mask.values()])

def sparsity_2(mask):
    
    numel=sum([torch.numel(tens) for tens in mask])
    num_ones=sum([torch.bincount(torch.flatten(tens.to(torch.int32)))[0] for tens in mask])

    return ((num_ones/numel).item())*100

def get_metrics(mask):

    return (sparsity(mask),importance(mask),localisation(mask))


def plot_metrics():
    network_names=['SCL','MRNet','Wild-ResNet','CNN-LSTM']
    task_names=['Squares','Circles','Triangles','Pi Rotation','2pi rotation','Black','White','Large','Small']

    network_name_colour={network_names[i]:i for i in range(len(network_names))}
    task_name_colour={task_names[i]:i for i in range(len(task_names))}

    masks=generate_multiple_masks(1000)
    mask_names=[random.choice(network_names) for mask in masks]
    metrics=[get_metrics(mask) for mask in masks]


    #plot

    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter3D(
        xs=[_[0] for _ in metrics],
        ys=[_[1] for _ in metrics],
        zs=[_[2] for _ in metrics],
        c=list(map(network_name_colour.get,mask_names)),
        alpha=0.5,
        s=30,
    )

    ax.set_xlabel('Importance');ax.set_xlim([0,1])
    ax.set_ylabel('Sparsity');ax.set_ylim([0,1])
    ax.set_zlabel('Localisation',fontsize=10);ax.set_zlim([0,1])
    ax.zaxis.labelpad=0
    ax.legend()
    fig.tight_layout()




def num_model_parameters(model):
    num_params=0
    for p in model.parameters():
        num_params+=(torch.prod(torch.tensor(p.shape))).item()
    return num_params


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_SCL_state_dict(path,train_wrapper=True):
    with open(path,'rb') as f:
        contents=CPU_Unpickler(f).load()

    state_dict=contents['model state dict']
    #assume SCL trained with training wrapper. Must edit key names
    values=list(state_dict.values())
    keys=list(state_dict.keys())
    new_keys=[s.lstrip('scl.') for s in keys]

    state_dict=dict(zip(new_keys,values))

    return state_dict

