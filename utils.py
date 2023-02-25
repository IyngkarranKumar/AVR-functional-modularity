#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn.functional as F
import random

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


            fig,axs = plt.subplots(4,6,figsize = (10,10))
            matrix_axs = [(0,0),(0,1),(0,2),
                            (1,0),(1,1),(1,2),
                            (2,0),(2,1),(2,2),
                            (0,4),(0,5),
                            (1,4),(1,5),
                            (2,4),(2,5),
                            (3,4),(3,5)] 

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

                ax.imshow(img)

            fig.suptitle(name,fontsize=15)
            fig.tight_layout()

def imshow(tens):
    arr=tens.permute(1,2,0).numpy()
    plt.imshow(arr)

def calculate_accuracy(pred_logits,targets):
    probs=F.softmax(pred_logits,dim=1)
    pred=torch.argmax(probs,axis=1)
    acc=(len(torch.nonzero(pred==targets))/len(pred))*100
    return acc

def sparsity(binary_mask_iter):
    total=sum(torch.numel(b) for b in binary_mask_iter)
    ones=sum(torch.count_nonzero(b) for b in binary_mask_iter)
    sparsity=(total-ones)/total
    return sparsity.item()


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


        U1 = torch.rand(1, requires_grad=True)
        U2 = torch.rand(1, requires_grad=True)

        samples={}
        for k,v in logit_dict.items():
            samples[k]=torch.sigmoid((v - torch.log(torch.log(U1) / torch.log(U2))) / tau)
            

        binaries_stop={}
        for k,v in samples.items():
            with torch.no_grad():
                binaries_stop[k]=(v>0.5).float()-v
        
        binaries={}
        for k,v in binaries_stop.items():
            binaries[k]=v+samples[k]

        return binaries



def generate_random_mask(sizes):

    weight_mask=[]
    for sz in sizes:
        t=torch.rand(sz)
        t[t<0.5]=0
        t[t>=0.5]=1
        weight_mask.append(t.int())

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
        


    return masks


def sparsity(mask):
    
    numel=sum([torch.numel(tens) for tens in mask])
    num_ones=sum([torch.bincount(torch.flatten(tens))[0] for tens in mask])

    return (num_ones/numel).item()

def importance(mask):

    return random.random()

def localisation(mask):
    
    return random.random()


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






