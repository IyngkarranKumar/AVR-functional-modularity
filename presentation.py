
import utils
import random
import matplotlib.pyplot as plt
import numpy as np




#datasets
if 0:
    utils.view_matrices(source_dir='datasets/pentagon')


#metrics 
if 1:
    
    network_names=['SCL','MRNet','Wild-ResNet','CNN-LSTM']
    task_names=['Squares','Circles','Triangles','Pi Rotation','2pi rotation','Black','White','Large','Small']

    network_name_colour={network_names[i]:i for i in range(len(network_names))}
    task_name_colour={task_names[i]:i for i in range(len(task_names))}

    masks=utils.generate_multiple_masks(50)
    mask_names=[random.choice(network_names) for mask in masks]
    metrics=[utils.get_metrics(mask) for mask in masks]


    #plot

    fig=plt.figure()
    ax=fig.add_subplot(projection='3d')
    ax.scatter3D(
        xs=[_[0] for _ in metrics],
        ys=[_[1] for _ in metrics],
        zs=[_[2] for _ in metrics],
        c=list(map(network_name_colour.get,mask_names)),
        alpha=1,
        s=60,
    )

    ax.set_xlabel('Importance');ax.set_xlim([0,1])
    ax.set_ylabel('Sparsity');ax.set_ylim([0,1])
    ax.set_zlabel('Localisation',fontsize=10);ax.set_zlim([0,1])
    ax.zaxis.labelpad=0
    fig.tight_layout()
