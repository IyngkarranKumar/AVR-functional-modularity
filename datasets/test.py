
# to generate sample dataset
#python I-RAVEN/main.py --num-samples 3 --save-dir ../save_dir_1 


#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import random


#%% generate

#python SRAN/I-RAVEN/main.py --num-samples <number of samples per configuration> --save-dir <directory to save the dataset>

#%%


def display_matrices(source_dir,dest_dir,n_imgs = 1):

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    else: #start again
        shutil.rmtree(dest_dir)

    
    subdirs = sorted(os.listdir(source_dir))

    #iterate through and generate images
    for subdir in subdirs:

        if not os.path.exists(os.path.join(dest_dir,subdir)):
            os.mkdir(os.path.join(dest_dir,subdir))

        file_names = sorted(os.listdir(os.path.join(source_dir,subdir))); 
        n_files = int(len(file_names) / 2)
        img_idxs = [2*idx for idx in random.sample(range(n_files),n_imgs)]


        for img_idx in img_idxs:
            img_file = file_names[img_idx];print(img_file)
            data = np.load(os.path.join(source_dir,subdir,img_file))
            target = data['target']
            images = data['image']; img_shape = images.shape[1:]

            name = subdir + '/' + os.path.splitext(img_file)[0]; print(name) 

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
        
            #save
            
            cwd = os.getcwd()
            save_path = os.path.join(cwd,dest_dir,name)
            plt.savefig(save_path,)

    

#%% dev code

if 0: #matrix viewing
    source_dir = 'save_dir_1'; subdir = 'center_single'; img_file = 'RAVEN_0_train.npz'

    data = np.load(os.path.join(source_dir,subdir,img_file))
    target = data['target']
    images = data['image']; img_shape = images.shape[1:] 

    fig,axs = plt.subplots(4,6,figsize = (10,10))

    #prepare axes
    for i in range(len(axs)):
        for j in range(len(axs[i])):
            ax = axs[i,j]
            ax.tick_params(left=False,right=False,labelleft=False,
                labelbottom=False,bottom=False)
            if not [(x,y) for x,y in matrix_axs if x==i and y==j]:
                ax.axis('off')
            


    matrix_axs = [(0,0),(0,1),(0,2),
                    (1,0),(1,1),(1,2),
                    (2,0),(2,1),(2,2),
                    (0,4),(0,5),
                    (1,4),(1,5),
                    (2,4),(2,5),
                    (3,4),(3,5)] 

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

        #select axes
        ax.imshow(img)
        
