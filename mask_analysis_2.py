import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt;plt.style.use('ggplot')
import seaborn as sns
import copy
import pickle
import os
import utils
import random
import importlib
import torch.nn.functional as F
import matplotlib as mpl

from utils import CPU_Unpickler

importlib.reload(utils)


#auxilliary functions


def sparsity(mask,logits=False):
    if logits:
        binary_mask=utils.transform_logit_tensors(mask)
        binary_mask_iter=binary_mask.values()
    else:
        binary_mask_iter=mask.values()
    total=sum(torch.numel(b) for b in binary_mask_iter)
    ones=sum(torch.count_nonzero(b) for b in binary_mask_iter)
    sparsity=(total-ones)/total
    return sparsity.item()*100

def localisation(mask,logits=False):

    if logits:
        binary_mask=utils.transform_logit_tensors(mask)
        binary_mask_iter=binary_mask.values()
    else:
        binary_mask_iter=mask.values()

    def normed_var(mask_tens):
        d=len(mask_tens.size())
        mask_idxs=mask_tens.nonzero().to(torch.float32)
        n=len(mask_idxs)
        if n==0:
            return 0
        mu=torch.mean(mask_idxs,dim=0)

        normed_var=(1/n*d)*sum([(F.mse_loss(idx,mu).item())/(torch.dot(mu,mu).item()+1e-12) for idx in mask_idxs])
        return normed_var
    
    normed_vars=[normed_var(mask_tens) for mask_tens in binary_mask_iter]

    return sum(normed_vars)


def mask_num_ones(mask):
    num_ones=sum([torch.bincount(torch.flatten(tens.to(torch.int32)),minlength=2)[-1] for tens in mask.values()])
    return num_ones.item()

class MaskAnalysis:


    def __init__(self,ckpt_paths,accuracies=[60,70,80],device='cpu',preprocessing=True):

        self.mask_logit_dict={}
        self.mask_binaries_dict={}
        self.device=torch.device(device)
        self.accuracies=accuracies
        self.preprocessing_dir=[path for path in ckpt_paths if 'preprocessing' in path][0]


        for i,acc_dir in enumerate(ckpt_paths):
            if 'preprocessing' in acc_dir: continue
            for task_name in os.listdir(acc_dir):
                key_name=f'SCL_{accuracies[i]}_{task_name}'
                task_dir=os.path.join(acc_dir,task_name)


                #GETS FIRST .CKPT FILE FOR NOW - need to modify to get chosen alpha
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


        if preprocessing:
            self.preprocessing()


    def preprocessing(self):

        #find modal masks and remove shared weights
        true_binaries_dict={}
        for acc_idx,acc in enumerate(self.accuracies):

            preprocessing_mask_path=[path for path in os.listdir(self.preprocessing_dir) if str(acc) in path][0]
            with open (os.path.join(self.preprocessing_dir,preprocessing_mask_path),'rb') as f:
                preprocessing_logit_dict=pickle.load(f)['logit_tensors_dict']
                preprocessing_binaries_dict=utils.transform_logit_tensors(preprocessing_logit_dict)



            acc_true_binaries_dict={}
            acc_mask_binaries_dict={k:v for k,v in self.mask_binaries_dict.items() if str(acc) in k}


            '''
            if sanity_check:
                for idx,layer in enumerate(layer_names):
                    if idx==0:
                        preprocessing_binaries_dict[layer]=torch.ones_like(next(iter(acc_mask_binaries_dict.values()))[layer])
                        preprocessing_binaries_dict[layer][0][0]=0
                    else:
                        preprocessing_binaries_dict[layer]=torch.zeros_like(next(iter(acc_mask_binaries_dict.values()))[layer])
            '''
            
            shared_weights={}



        #find shared weights across all layers
            for idx,tens_name in enumerate(self.layer_names):
                
                tensor_list=[mask[tens_name].to(torch.int32) for mask in acc_mask_binaries_dict.values()] #get binary tensors for this name

                #find shared
                tens_shared=(torch.ones(size=tensor_list[0].size(),dtype=torch.int32))
                for t in tensor_list:
                    tens_shared=torch.logical_and(tens_shared,t)
                tens_shared=tens_shared.int()


                shared_weights[tens_name]=tens_shared.int()

            #not returned, but useful info
            num_ones_masks=[mask_num_ones(mask) for mask in acc_mask_binaries_dict.values()]
            num_ones_shared=mask_num_ones(shared_weights)
            max_shared_perc=(num_ones_shared/min(num_ones_masks))*100

            #remove shared weights
            for mask_name,mask in acc_mask_binaries_dict.items():
                if 'preprocessing' in mask_name: continue #we don't care about true weights for preprocessing mask
                masks_iter=list(zip(mask.values(),shared_weights.values()))
                name_masks_iter = list(zip(self.layer_names,masks_iter))
                acc_true_binaries_dict[mask_name]={k:(v0-v_shared).to(torch.int32) for k,(v0,v_shared) in name_masks_iter}


            true_binaries_dict={**true_binaries_dict,**acc_true_binaries_dict}
        self.mask_binaries_dict=true_binaries_dict



    def line_plots(self):

        accuracies=self.accuracies
        shapes2process=['circles','triangles','squares','pentagon','hexagon']

        all_sparsities=[]
        all_localisations=[]

        for shape in shapes2process:
            shape_sparsities=[]
            shape_localisations=[]

            for acc in accuracies:
                key_name=f'SCL_{acc}_{shape}'
                mask=self.mask_binaries_dict[key_name]
                shape_sparsities.append(sparsity(mask))
                shape_localisations.append(localisation(mask))
            all_sparsities.append(shape_sparsities)
            all_localisations.append(shape_localisations)



        #plot
        fig,(ax_s,ax_l)=plt.subplots(nrows=2,sharex=True,figsize=(10,6))
        axs=(ax_s,ax_l)

        marker=11
        marker_size=15

        x_ticks=np.arange(len(accuracies))
        x_names=[f'SCL_{acc}' for acc in accuracies]

        for i in range(len(shapes2process)):
            ax_s.plot(x_ticks,all_sparsities[i],label=shapes2process[i],marker=marker,markersize=marker_size)
            ax_l.plot(x_ticks,all_localisations[i],label=shapes2process[i],marker=marker,markersize=marker_size)


        #beautify


        ax_s.legend(loc=(1.05,0),fontsize=20)
        ax_s.set_ylabel('Sparsity',fontsize=20)
        ax_l.set_ylabel('Localisation',fontsize=20)
        ax_l.set_xticks(x_ticks,[str(acc) for acc in accuracies],fontsize=20)
        ax_l.set_xlabel('SCL model accuracies',fontsize=20)
        for ax in axs:
            ax.tick_params(axis='y',labelsize=15)

        fig.tight_layout()



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


        ax.bar(x_pos,y_vals,width=1.0)


        #beautification
        tick_size=15
        label_size=15
        ax.set_yscale('log')
        plt.xticks(x_pos,x_names,rotation=45,fontsize=8)
        plt.yticks(fontsize=tick_size)
        plt.ylabel('$\log(n_i / n_{tot})$',fontsize=label_size)
        plt.xlabel('SCL Layers',fontsize=label_size,labelpad=30)
        plt.title(f'Model accuracy:{acc} %',fontsize=20)

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
        fig.suptitle(f'Model accuracy: {acc}%',fontsize=20)
        _l=list(acc_mask_binaries_dict.keys())
        x_labels=y_labels=[s.split('_')[-1] for s in _l]

        #beautification
        fmt='.1f' #use to get rid of standard form
        cmap='magma_r'

        plot_kwargs={
            'data':plot_matrix,
            'vmax':100,
            'vmin':0,
            'annot':True,
            'annot_kws':{'fontsize':12},
            'ax':ax,
            'xticklabels':x_labels,
            'yticklabels':y_labels,
            'fmt':fmt,
            'cmap':cmap
        }


        ax_mat=sns.heatmap(**plot_kwargs)
        ax_mat.tick_params(axis='both',labelsize=15)
        ax_mat.tick_params('x',rotation=45)
        ax_mat.tick_params('y',rotation=0)

        cbar=ax_mat.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)


#test
if 1:

    ckpt_paths=['masks/SCL_80','masks/SCL_70','masks/SCL_60','masks/preprocessing_masks']

    mask_analysis=MaskAnalysis(ckpt_paths=ckpt_paths,preprocessing=True)

    mask_logit_dict=mask_analysis.mask_logit_dict
    mask_binaries_dict=mask_analysis.mask_binaries_dict
    layer_names=mask_analysis.layer_names

    if 1:
        mask_analysis.spatial_distribution_plots(acc=80)

    if 1:
        mask_analysis.sharing_matrices(acc=80)

    if 1:
        mask_analysis.line_plots()

