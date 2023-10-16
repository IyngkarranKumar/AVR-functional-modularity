import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl;
import importlib
import matplotlib.pyplot as plt
import numpy as np
import data
import utils
import sys
import wandb
import copy
import pickle

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from abc import ABC,abstractmethod
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from torch.special import logit
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from timeit import default_timer as timer

importlib.reload(data)
importlib.reload(utils)

debug=False

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children



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


class AbstractMaskedModel(ABC):

    def __init__(self,model,train_dataloader,test_dataloader1,test_dataloader2,device,savedir=None,logit_init=0.9):
        
        self.model=model
        self.train_dataloader=train_dataloader
        self.test_dataloader1=test_dataloader1
        self.test_dataloader2=test_dataloader2
        self.device=device
        self.model.to(self.device)
        #freeze model parameters
        for p in model.parameters():
            p.requires_grad=False
        self.param_dict=dict(model.named_parameters())
        self.leaf_modules=utils.get_named_children(self.model)
        self.savedir=savedir


        self.logit_tensors_dict={k:torch.nn.Parameter(data=torch.full_like(p,logit_init,device=self.device)) for k,p in model.named_parameters()}
        self.alpha=None
        self.tau=None
        self.lr=None
        self.logging=False #this attribute and below are set during training/loading
        self.logger=None
        self.log_dict=None
        self.run_id=None
        self.optimiser=None

        self.binaries=None; self.transform_logit_tensors() #initiaise binaries

        self.global_step=0
        self.train_epoch=0
        self.early_stopping=None
        self.ES_accuracies=[] #accuracies for early stopping

    @abstractmethod
    def forward(self,x,invert_mask=False):
        pass

    def calculate_loss(self,y_hat,y):
        crossent_loss=F.cross_entropy(y_hat,y)
        reg_loss=self.alpha*torch.sum(torch.stack([torch.sum(logit_tens) for logit_tens in list(self.logit_tensors_dict.values())]))
        loss=crossent_loss+reg_loss
        acc=utils.calculate_accuracy(y_hat,y)

        return crossent_loss,reg_loss,loss,acc

    def train(self,alpha,tau=1,n_epochs=5,lr=1e-3,n_batches=5,batch_split=4,
                    val_every_n_steps=10,n_val_batches=100,
                    eval_every_n_steps=10,n_eval_batches=5,
                    logging=False,set_log_name=False,save_freq_epoch=1e10,save_freq_step=1e10,early_stopping=None,sweep=False,sweep_logger=None):


            #set class attributes for use in rest of class
            self.alpha=alpha
            self.tau=tau
            self.lr=lr
            self.optimiser=torch.optim.Adam(self.logit_tensors_dict.values(),lr=self.lr)
            self.early_stopping=early_stopping
            
            if logging:
                self.logging=True
            if self.logging:
                if set_log_name:
                    log_name=str(input('Enter log name'))
                    if log_name=='':
                        sys.exit()
                else:
                    log_name=None
                
                if self.run_id is not None:
                    self.logger=wandb.init(id=self.run_id,project='AVR',resume='must')
                else:
                    if sweep:
                        self.logger=sweep_logger
                    if not sweep:
                        log_name=str(input('Enter log name'))
                        self.logger=wandb.init(project='AVR',name=log_name)
                    

                self.log_dict={}

            for epoch in range(self.train_epoch,n_epochs):
                start_time=timer()
                for batch_idx,batch in enumerate(self.train_dataloader):
                    #print(f'Starting train batch {batch_idx}')
                    if n_batches=='full':
                        pass
                    if batch_idx==n_batches:
                        break


                    train_loss=train_crossent_loss=train_reg_loss=0
                    self.optimiser.zero_grad()

                    x,y,*rest=batch
                    x,y=x.to(self.device),y.to(self.device)
                    split_X,split_y=torch.chunk(x,batch_split),torch.chunk(y,batch_split)

                    logits=[]
                    for x_,y_ in zip(split_X,split_y):
                        y_hat=self.forward(x_)
                        logits.append(y_hat)
                        crossent_loss,reg_loss,loss,_=self.calculate_loss(y_hat,y_)
                        train_loss+=loss.item()
                        train_crossent_loss+=crossent_loss.item()
                        train_reg_loss+=reg_loss.item()
                        loss.backward() #accumulate losses
                    train_acc=utils.calculate_accuracy(torch.concatenate(logits),y) #calculate accuracy over whole batch, rather than sub-batches

                    self.optimiser.step()

                    if self.logging:
                        train_log_dict={
                            'Loss/train':train_loss,
                            'Loss/train_cross_entropy':train_crossent_loss,
                            'Loss/train_reg':train_reg_loss,
                            'Accuracy/train':train_acc,
                        }
                        self.log_dict.update(train_log_dict)

                    #val
                    if (self.global_step%val_every_n_steps==0):
                        self.validation(n_batches=n_val_batches)

                    #ablation
                    if (epoch%eval_every_n_steps==0) and (epoch!=0):
                        self.eval(self.test_dataloader1,self.test_dataloader2,n_batches=n_eval_batches)
                        end_eval_time=timer()

                    #sparsities
                    sparsity=utils.sparsity(self.binaries)
                    if self.logging:
                        self.log_dict['Sparsity']=sparsity


                    #logging
                    if self.logging:
                        wandb.log(self.log_dict)



                    self.global_step+=1
                    







                
                #save every n_save epochs
                if (self.savedir is not None) and (epoch%save_freq_epoch==0) and (epoch!=0):
                    self.save()

                #save on last epoch
                if (self.savedir is not None) and (epoch==n_epochs-1):
                    self.save()


                if (self.savedir is not None) and (self.global_step%save_freq_step==0) and (self.global_step!=0):
                    self.save()

                    
                print(f'Epoch: {epoch}, Loss:{train_loss}')
                self.train_epoch+=1
                
            
            wandb.finish()
            print('Training finished')
                
    def validation(self,n_batches):
        batch=next(iter(self.test_dataloader1))

        losses1=[]
        val_accs1=[]
        losses2=[] #2 index refers to not-task
        val_accs2=[]

        for batch_idx,(batch_task,batch_not_task) in enumerate(zip(self.test_dataloader1,self.test_dataloader2)):
            #print(f'Starting validation batch {batch_idx}')
            if n_batches=='full':
                pass
            if batch_idx==n_batches:
                break

            x1,y1,*rest1=batch_task
            x2,y2,*rest2=batch_not_task
            x1,y1,x2,y2=x1.to(self.device),y1.to(self.device),x2.to(self.device),y2.to(self.device)

            with torch.no_grad():
                y_hat1=self.forward(x1)
                crossent_loss1,reg_loss1,loss1,acc1=self.calculate_loss(y_hat1,y1)
                losses1.append((crossent_loss1.item(),reg_loss1.item(),loss1.item()))
                val_accs1.append(acc1)

                y_hat2=self.forward(x2)
                crossent_loss2,reg_loss2,loss2,acc2=self.calculate_loss(y_hat2,y2)
                losses2.append((crossent_loss2.item(),reg_loss2.item(),loss2.item()))
                val_accs2.append(acc2)

            val_crossent_loss1=np.mean([_[0] for _ in losses1])
            val_reg_loss1=np.mean([_[1] for _ in losses1])
            val_loss1=np.mean([_[2] for _ in losses1])
            val_accuracy1=np.mean(val_accs1)

            val_crossent_loss2=np.mean([_[0] for _ in losses2])
            val_reg_loss2=np.mean([_[1] for _ in losses2])
            val_loss2=np.mean([_[2] for _ in losses2])
            val_accuracy2=np.mean(val_accs2)
            

        if self.logging:
            self.log_dict.update({
                            'Loss/validation_task':val_loss1,
                            'Loss/validation_cross_entropy_task':val_crossent_loss1,
                            'Loss/validation_reg_task':val_reg_loss1,
                            'Accuracy/validation_task':val_accuracy1,
                            'Loss/validation_not_task':val_loss2,
                            'Loss/validation_cross_entropy_not_task':val_crossent_loss2,
                            'Loss/validation_reg_not_task':val_reg_loss2,
                            'Accuracy/validation_not_task':val_accuracy2
                            })

    def eval(self,task_eval_dataloader,_task_eval_dataloader,n_batches):
        '''
        Evaluated mask via ablation

        Ablation - frozen_parameters * ~binaries (inverted mask)
        '''
        #create masked model

        acc1s=[]
        acc2s=[]


        for batch_idx,(batch1,batch2) in enumerate(zip(task_eval_dataloader,_task_eval_dataloader)):
            if n_batches=='full':
                pass
            if batch_idx==n_batches:
                break
            x1,y1,*rest=batch1
            x2,y2,*rest=batch2
            x1,y1=x1.to(self.device),y1.to(self.device)
            x2,y2=x2.to(self.device),y2.to(self.device)


            pred_logits_1=self.forward(x1,invert_mask=True)
            pred_logits_2=self.forward(x2,invert_mask=True)

            acc1s.append(utils.calculate_accuracy(pred_logits_1,y1))
            acc2s.append(utils.calculate_accuracy(pred_logits_2,y2))



        acc1=round(np.mean(acc1s),2)
        acc2=round(np.mean(acc2s),2)

        if self.logging:
            self.log_dict.update({'Ablation/Task':acc1,'Ablation/NOT Task':acc2})
        else:
            print({"Acc task'":acc1,"Acc not task":acc2})


    def MaskedLinear(self,x,name,invert=False):



        binary_weight,binary_bias=self.binaries[name+'.weight'],self.binaries[name+'.bias']
        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()

        masked_weight,masked_bias=self.param_dict[name+'.weight']*binary_weight,self.param_dict[name+'.bias']*binary_bias
        out=F.linear(x,weight=masked_weight,bias=masked_bias)
        return out

    def MaskedConv2d(self,x,name,bias=True,invert=False):

        '''
        invert detaches tensor from comp graph, so should only be used during val
        '''

        stride,padding=self.leaf_modules[name].stride,self.leaf_modules[name].padding

        binary_weight=self.binaries[name+'.weight']

        if bias:
            binary_bias=self.binaries[name+'.bias']
        else:
            masked_bias=None

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            if bias:
                binary_bias=(~(binary_bias.bool())).int()

        masked_weight=self.param_dict[name+'.weight']*binary_weight
        if bias:
            masked_bias=self.param_dict[name+'.bias']*binary_bias
        else:
            masked_bias=None
        out=F.conv2d(x,weight=masked_weight,bias=masked_bias,stride=stride,padding=padding)
        return out

    def MaskedBatchNorm2d(self,x,name,invert=False):
        
        #these are approximations to feature mean + variance over whole dataset, calculated during training
        running_mean=self.leaf_modules[name].running_mean
        running_var=self.leaf_modules[name].running_var 

        binary_weight=self.binaries[name+'.weight']
        binary_bias=self.binaries[name+'.bias']

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()

        
        masked_weight=self.param_dict[name+'.weight']*binary_weight
        masked_bias=self.param_dict[name+'.bias']*binary_bias
        return F.batch_norm(x,running_mean=running_mean,running_var=running_var,weight=masked_weight,bias=masked_bias)

    def MaskedLayerNorm(self,x,name,invert=False):

        normalized_shape=self.leaf_modules[name].normalized_shape

        binary_weight=self.binaries[name+'.weight']
        binary_bias=self.binaries[name+'.bias']

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()
        
        masked_weight=self.param_dict[name+'.weight']*binary_weight
        masked_bias=self.param_dict[name+'.bias']*binary_bias
        
        

        return F.layer_norm(x,normalized_shape=normalized_shape,weight=masked_weight,bias=masked_bias)

    def transform_logit_tensors(self,return_binaries=False):


        samples={}
        for k,v in self.logit_tensors_dict.items():
            U1=torch.rand_like(v,requires_grad=True).to(self.device)
            U2=torch.rand_like(v,requires_grad=True).to(self.device)
            samples[k]=torch.sigmoid((v - torch.log(torch.log(U1) / torch.log(U2))) / (self.tau if self.tau is not None else 1))
            

        binaries_stop={}
        for k,v in samples.items():
            with torch.no_grad():
                binaries_stop[k]=(v>0.5).float()-v
        
        binaries={}
        for k,v in binaries_stop.items():
            binaries[k]=v+samples[k]

        if not return_binaries:
            self.binaries=binaries
        else:
            return binaries

    def save(self):

        name=f'alpha={self.alpha}_checkpoint_step={self.global_step}_epoch={self.train_epoch}'

        if not os.path.isdir(self.savedir):
            os.makedirs(self.savedir)

        save_dict={}
        save_dict['alpha']=self.alpha
        save_dict['tau']=self.tau
        save_dict['global_step']=self.global_step
        save_dict['train_epoch']=self.train_epoch
        save_dict['logit_tensors_dict']=self.logit_tensors_dict
        save_dict['optimiser']=self.optimiser
        if self.logging:
            save_dict['run_id']=self.logger.id
        else:
            save_dict['run_id']=None

        fname=os.path.join(self.savedir,name)
        with open(fname,'wb') as f:
            pickle.dump(save_dict,f)
            print(f'step={self.global_step}, epoch={self.train_epoch} saved')
 
    def load(self,path):

        with open (path,'rb') as f:
            load_dict=pickle.load(f)

        self.global_step=load_dict.get('global_step')
        self.train_epoch=load_dict.get('train_epoch')+1
        self.logit_tensors_dict=load_dict.get('logit_tensors_dict')
        self.optimiser=load_dict.get('optimiser')
        self.run_id=load_dict.get('run_id')
    

class MaskedMNISTFFN(AbstractMaskedModel):

    def __init__(self,kwargs):
        super().__init__(**kwargs)

        #none mask trainable layers
        self.layer0=nn.Flatten(start_dim=1,end_dim=-1)

    def forward(self, x, invert_mask=False):
        self.transform_logit_tensors()
        
        
        x0=self.layer0(x)
        x1=self.MaskedLinear(x0,name='layers.1',invert=invert_mask)
        x2=self.MaskedLinear(x1,name='layers.2',invert=invert_mask)
        x3=self.MaskedLinear(x2,name='layers.3',invert=invert_mask)
        x4=self.MaskedLinear(x3,name='layers.4',invert=invert_mask)

        return x4

class MaskedMNISTConv(AbstractMaskedModel):
    
    def __init__(self,kwargs):
        super().__init__(**kwargs)


        #initialise layers that mask not trained on
        #should implement method to check if we've done this right
        self.maxpool_2=nn.MaxPool2d(kernel_size=(2,2))
        self.conv2_drop=nn.Dropout()

    def forward(self,x,invert_mask=False):
        self.transform_logit_tensors()

        N=x.size()[0]
        
        x=F.relu(self.maxpool_2(self.MaskedConv2d(x,name='conv1',invert=invert_mask)))
        x=F.relu(self.maxpool_2(self.conv2_drop(self.MaskedConv2d(x,name='conv2',invert=invert_mask))))
        x=x.view(N,-1)
        x=F.relu(self.MaskedLinear(x,name='fc1',invert=invert_mask))
        x=F.dropout(x)
        x=self.MaskedLinear(x,name='fc2',invert=invert_mask)
        return x


class MaskedSCLModel(AbstractMaskedModel):

    def __init__(self,kwargs):
        super().__init__(**kwargs)

        self.flatten_layer_vision=nn.Flatten(1)
        self.relu=nn.ReLU(inplace=True)

    def MaskedVisionNet(self,x,invert_mask=False):

        '''
        vision_module_names=[_ for _ in self.leaf_modules.keys() if 'vision' in _]
        vision_modules={k:self.leaf_modules[k] for k in vision_module_names}

        for name,module in vision_modules.items():
            if isinstance(module,nn.Conv2d):
                x=self.MaskedConv2d(x,name=name,bias=True,invert=invert_mask)
            elif isinstance(module,nn.Linear):
                x=self.MaskedLinear(x,name=name,invert=invert_mask)
            elif isinstance(module,nn.BatchNorm2d):
                x=self.MaskedConv2d(x,name=name,invert=invert_mask)
            elif isinstance(module,nn.LayerNorm):
                pass
            elif isinstance(module,nn.Flatten):
                pass
            elif isinstance(module,nn.ReLU):
                pass
            else:
                raise Exception('Unrecognised module')
        '''


        x=self.MaskedConv2d(x,name='vision.net.0',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='vision.net.1',invert=invert_mask)

        x=self.MaskedConv2d(x,name='vision.net.2',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='vision.net.3',invert=invert_mask)

        x=self.MaskedConv2d(x,name='vision.net.4',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='vision.net.5',invert=invert_mask)

        x=self.MaskedConv2d(x,name='vision.net.6',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='vision.net.7',invert=invert_mask)

        x_conv_out=self.MaskedConv2d(x,name='vision.net.8',bias=True,invert=invert_mask)

        x1=self.flatten_layer_vision(x_conv_out)
        x1=self.MaskedLinear(x1,name='vision.net.10',invert=invert_mask)
        x1=self.relu(x1)

        #feedforward residual layer
        x2=self.MaskedLinear(x1,name='vision.net.12.net.0',invert=invert_mask)
        x2=self.MaskedLayerNorm(x2,name='vision.net.12.net.1',invert=invert_mask)
        x2=self.relu(x2)
        x2=self.MaskedLinear(x2,name='vision.net.12.net.3',invert=invert_mask)
        out=x2+x1
        
        return out

    def MaskedAttrNet(self,x,invert_mask=False):
        
        shape,heads=x.shape,self.model.attr_heads
        dim=shape[-1]

        #scattering transform
        x=x.reshape(-1,heads,dim // heads)
        x=self.MaskedLinear(x,name='attr_net.mlp.net.0',invert=invert_mask)
        x=self.relu(x)
        x=self.MaskedLinear(x,name='attr_net.mlp.net.2',invert=invert_mask)
        x=x.reshape(shape)

        #feed forward residual net
        x1=self.MaskedLinear(x,name= 'ff_residual.net.0',invert=invert_mask)
        x1=self.MaskedLayerNorm(x1,name= 'ff_residual.net.1',invert=invert_mask)
        x1=self.relu(x1)
        x1=self.MaskedLinear(x1,name= 'ff_residual.net.3',invert=invert_mask)
        out=x+x1

        return out


    def MaskedRelNet(self,x,invert_mask=False):

        x=self.MaskedLinear(x,name= 'rel_net.net.0',invert=invert_mask)
        x=self.relu(x)
        x=self.MaskedLinear(x,name= 'rel_net.net.2',invert=invert_mask)
        x=self.relu(x)
        x=self.MaskedLinear(x,name= 'rel_net.net.4',invert=invert_mask)

        return x

    def MaskedToLogit(self,x,invert_mask=False):

        x=self.MaskedLinear(x,name='to_logit',invert=invert_mask)
        return x


    def preprocess(self,x):

        questions,answers=x[:,0:8:,:,:,].unsqueeze(2),x[:,8:,:,:,].unsqueeze(2)
        answers=answers.unsqueeze(2)
        questions=utils.expand_dim(questions, dim=1, k=8)
        permutations=torch.cat((questions, answers), dim=2)

        return permutations


    def forward(self, x, invert_mask=False):
        self.transform_logit_tensors() #update binaries

        x=self.preprocess(x)
        b,m,n,c,h,w=x.shape
        x=x.view(-1,c,h,w)


        features=self.MaskedVisionNet(x,invert_mask=invert_mask)

        attrs=self.MaskedAttrNet(features,invert_mask=invert_mask)
        attrs=attrs.reshape(b,m,n,self.model.rel_heads,-1).transpose(-2,-3).flatten(3)

        rels=self.MaskedRelNet(attrs,invert_mask=invert_mask)
        rels=rels.flatten(2)

        logits=self.MaskedToLogit(rels).flatten(1)
    

        return logits

class MaskedCNN_MLP(AbstractMaskedModel):

    def __init__(self,kwargs):
        super().__init__(**kwargs)

    def MaskedConvModule(self,x,invert_mask=False):

        x=self.MaskedConv2d(x,name='conv.conv1',invert=invert_mask)
        x=F.relu(self.MaskedBatchNorm2d(x,name='conv.batch_norm1',invert=invert_mask))
        x=self.MaskedConv2d(x,name='conv.conv2',invert=invert_mask)
        x=F.relu(self.MaskedBatchNorm2d(x,name='conv.batch_norm2',invert=invert_mask))
        x=self.MaskedConv2d(x,name='conv.conv3',invert=invert_mask)
        x=F.relu(self.MaskedBatchNorm2d(x,name='conv.batch_norm3',invert=invert_mask))
        x=self.MaskedConv2d(x,name='conv.conv4',invert=invert_mask)
        x=F.relu(self.MaskedBatchNorm2d(x,name='conv.batch_norm4',invert=invert_mask))
        
        x_conv=x.view(x.size()[0],-1)

        return x_conv

    def forward(self,x,invert_mask=False):

        x=self.MaskedConvModule(x,invert_mask=invert_mask)
        x=F.relu(self.MaskedLinear(x,name='l1'))
        x=F.relu(self.MaskedLinear(x,name='l2'))

        return x