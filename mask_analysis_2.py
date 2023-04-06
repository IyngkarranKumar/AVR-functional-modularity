import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt;plt.style.use('ggplot')
import seaborn as sns
import copy
import pickle
import os
import utils
import importlib
import torch.nn.functional as F
import matplotlib as mpl

from utils import CPU_Unpickler

importlib.reload(utils)


#helper functions
def mask_num_ones(mask):
    num_ones=sum([torch.bincount(torch.flatten(tens.to(torch.int32)),minlength=2)[-1] for tens in mask.values()])
    return num_ones.item()

class MaskAnalysis:


    def __init__(self,ckpt_paths,accuracies=[80,70,60],device='cpu'):

        self.mask_logit_dict={}
        self.mask_binaries_dict={}
        self.device=torch.device(device)


        for i,acc_dir in enumerate(ckpt_paths):
            for task_name in os.listdir(acc_dir):
                key_name=f'SCL_{accuracies[i]}_{task_name}'
                task_dir=os.path.join(acc_dir,task_name)


                #GETS FIRST .CKPT FILE FOR NOW
                ckpt_file=os.path.join(task_dir,os.listdir(task_dir)[0])

                with open(ckpt_file,'rb') as f:
                    if device=='cpu':
                        data=CPU_Unpickler(f).load()
                    else:
                        data=pickle.load(f)

                    self.mask_logit_dict[key_name]=data['logit_tensors_dict']
                    self.mask_binaries_dict[key_name]=utils.transform_logit_tensors(data['logit_tensors_dict'])

        self.sanity_check_binaries_dict=copy.deepcopy(self.mask_binaries_dict)
        for mask in self.mask_binaries_dict:
            for idx,layer in enumerate(self.mask_binaries_dict[mask].keys()):
                if idx==0:
                    self.sanity_check_binaries_dict[mask][layer]=torch.ones_like(self.mask_binaries_dict[mask][layer])
                else:
                    self.sanity_check_binaries_dict[mask][layer]=torch.zeros_like(self.mask_binaries_dict[mask][layer])

                

        
        first_key=next(iter(self.mask_logit_dict))
        self.layer_names=self.mask_logit_dict[first_key].keys()






        



    def preprocessing():

        #find modal masks and remove shared weights
        pass


    def line_plots():
        pass

    def spatial_distribution_plots(self,acc,shape_type='all'):
        acc=int(acc)

        if shape_type=='all':
            shapes2process=['circles','triangles','squares','pentagon','hexagon']
        else:
            shapes2process=[shape_type]

        frac_means={}

        for idx,shape in enumerate(shapes2process):
            key_name=f'SCL_{acc}_{shape}'
            shape_binaries_dict=self.mask_binaries_dict[key_name]
            n_ones=utils.mask_num_ones(shape_binaries_dict)

            if idx==0:
                for layer in shape_binaries_dict.keys():
                    frac_means[layer]=utils.mask_num_ones(shape_binaries_dict[layer])/n_ones
            else:
                for layer in shape_binaries_dict.keys():
                    shape_frac=utils.mask_num_ones(shape_binaries_dict[layer])/n_ones
                    frac_means[layer]=(frac_means[layer]*idx)/(idx+1) + (shape_frac/(idx+1)) #update running mean


        #defo a far better 
        plot_layer_names={}
        plot_names=[]
        reg_exs=['vision','attr','ff_residual','rel','logit']
        reg_exs_repl=['V','A','FFR','R','L']

        for j in range(len(reg_exs)):
            _names=[l for l in layer_names if reg_exs[j] in l]
            _plot_names=[f'{reg_exs_repl[j]}{i+1}' for i in range(len(_names))]
            plot_names=plot_names+_plot_names


        x_names=plot_names;x_pos=5*np.arange(len(x_names)) 
        y_vals=list(frac_means.values())
        fig,ax=plt.subplots(figsize=(10,4));ax.set_yscale('log')

        tick_size=15
        label_size=15

        ax.bar(x_pos,y_vals,width=0.8)
        plt.xticks(x_pos,x_names,rotation=45,fontsize=8)
        plt.yticks(fontsize=tick_size)
        plt.ylabel('$\log(n_i / n_{tot})$',fontsize=label_size)

    def sharing_matrices(self,acc):
        acc=int(acc)

        acc_mask_binaries_dict={k:v for k,v in mask_binaries_dict.items() if str(acc) in k}

        shared_weights={}
        for idxa,maska_name in enumerate(acc_mask_binaries_dict):
            for idxb,maskb_name in enumerate(acc_mask_binaries_dict):
                if idxb<idxa:
                    continue
                maska=acc_mask_binaries_dict[maska_name]
                maskb=acc_mask_binaries_dict[maskb_name]
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
                
                shared_weights[f'{maska_name},{maskb_name}']=ab_shared_weights

        shared_names=shared_weights.keys()

        max_true_sharing_percentage={}
        for name in shared_names:
            maska,maskb=name.split(',')
            norm_const=min([mask_num_ones(acc_mask_binaries_dict[maska]),mask_num_ones(acc_mask_binaries_dict[maskb])])
            max_shared_perc_ab = round(((mask_num_ones(shared_weights[name]))/norm_const)*100,2)
            max_true_sharing_percentage[name]=max_shared_perc_ab

        n_masks=len(acc_mask_binaries_dict)
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
        x_labels=y_labels=list(acc_mask_binaries_dict.keys())
        fmt='.1f' #use to get rid of standard form
        cmap='magma_r'

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
        


#test
if 1:

    ckpt_paths=['masks/SCL_80','masks/SCL_70','masks/SCL_60']

    mask_analysis=MaskAnalysis(ckpt_paths=ckpt_paths)

    mask_logit_dict=mask_analysis.mask_logit_dict
    mask_binaries_dict=mask_analysis.mask_binaries_dict
    layer_names=mask_analysis.layer_names

    if 0:
        mask_analysis.spatial_distribution_plots(acc=80)

    if 1:
        mask_analysis.sharing_matrices(acc=80)



#line plots