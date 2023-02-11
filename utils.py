#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn.functional as F

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