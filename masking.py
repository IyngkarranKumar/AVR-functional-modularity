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
import importlib
import pickle
import wandb
import copy

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from copy import deepcopy
from torch.special import logit
from pytorch_lightning import loggers as pl_loggers
from torch.utils.tensorboard import SummaryWriter
from abc import ABC,abstractmethod

importlib.reload(data)
importlib.reload(utils)


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

    def __init__(self,model,train_dataloader,test_dataloader1,test_dataloader2,savedir=None):
        
        self.model=model
        self.train_dataloader=train_dataloader
        self.test_dataloader1=test_dataloader1
        self.test_dataloader2=test_dataloader2
        #freeze model parameters
        for p in model.parameters():
            p.requires_grad=False
        self.param_dict=dict(model.named_parameters())
        self.leaf_modules=utils.get_named_children(self.model)
        self.savedir=savedir


        self.logit_tensors_dict={k:torch.nn.Parameter(data=torch.full_like(p,0.9)) for k,p in model.named_parameters()}
        self.alpha=None
        self.logging=False #this attribute and below are set during training/loading
        self.logger=None
        self.run_id=None
        self.optimiser=torch.optim.Adam(self.logit_tensors_dict.values())

        self.global_step=0
        self.train_epoch=0

    @abstractmethod
    def forward(self,x,invert_mask=False):
        pass

    def calculate_loss(self,y_hat,y):
        crossent_loss=F.cross_entropy(y_hat,y)
        reg_loss=self.alpha*torch.sum(torch.stack([torch.sum(logit_tens) for logit_tens in list(self.logit_tensors_dict.values())]))
        loss=crossent_loss+reg_loss
        acc=utils.calculate_accuracy(y_hat,y)

        return crossent_loss,reg_loss,loss,acc

    def train(self,alpha,tau=1,n_epochs=5,n_batches=5,batch_split=4,
                    val_every_n_steps=10,n_val_batches=100,
                    eval_every=10,n_eval_batches=5,
                    logging=False,set_log_name=False,save_freq=10):


            #set class attributes for use in rest of class
            self.alpha=alpha
            self.tau=tau
            
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
                    self.logger=wandb.init(project='AVR',name=log_name)
                wandb.define_metric('global step')

            for epoch in range(self.train_epoch,n_epochs):
                start_time=timer()
                for batch_idx,batch in enumerate(self.train_dataloader):
                    if n_batches=='full':
                        pass
                    if batch_idx==n_batches:
                        break


                    train_loss=0
                    split_X,split_y=torch.chunk(batch[0],batch_split),torch.chunk(batch[1],batch_split)
                    for x,y in zip(split_X,split_y):
                        y_hat=self.forward(x)
                        crossent_loss,reg_loss,loss,acc=self.calculate_loss(y_hat,y)
                        train_loss+=loss.item()
                        loss.backward()

                    self.optimiser.step()

                    if self.logging:
                        '''
                        wandb.log({'epoch':epoch,
                                    'train_loss':train_loss,
                                    },step=self.global_step)
                        '''
                        wandb.define_metric('train_loss',step_metric='global_step')
                        wandb.log({'train_loss':train_loss,'global_step':self.global_step})

                    if (self.global_step%val_every_n_steps==0) and (self.global_step!=0):
                        self.validation(n_batches=n_val_batches)

                    self.global_step+=1
                    

                end_train_time=timer()



                #run ablation every n_ablation epochs
                if (epoch%eval_every==0) and (epoch!=0):
                    self.eval(self.test_dataloader1,self.test_dataloader2,n_batches=n_eval_batches)
                    end_eval_time=timer()

                    train_time=end_train_time-start_time
                    eval_time=end_eval_time-end_train_time
                    print(f'Train time: {train_time} \n Eval time:{eval_time}')

                
                #save every n_save epochs
                if (self.savedir is not None) and (epoch%save_freq==0) and (epoch!=0):
                    self.save()

                    
                print(f'Epoch: {epoch}, Loss:{loss.item()}')
                self.train_epoch+=1
                
            
            wandb.finish()
            print('Training finished')
                
    def validation(self,n_batches):
        batch=next(iter(self.test_dataloader1))

        losses=[]
        val_accs=[]

        for batch_idx,batch in enumerate(self.test_dataloader1):
            if n_batches=='full':
                pass
            if batch_idx==n_batches:
                break

            x,y=batch
            with torch.no_grad():
                y_hat=self.forward(x)
            crossent_loss,reg_loss,loss,acc=self.calculate_loss(y_hat,y)
            losses.append((crossent_loss.item(),reg_loss.item(),loss.item()))
            val_accs.append(acc)

        val_crossent_loss=np.mean([_[0] for _ in losses])
        val_reg_loss=np.mean([_[1] for _ in losses])
        val_loss=np.mean([_[2] for _ in losses])
        val_accuracy=np.mean(val_accs)

        if self.logging:
                wandb.log({
                            'validation_loss':val_loss,
                            'validation_crossent_loss':val_crossent_loss,
                            'validation_reg_loss':val_reg_loss,
                            'validation_accuracy':val_accuracy
                            })
        else:
            #print(f'\n Validation accuracy: {acc}')
            pass

        

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
            x1,y1=batch1
            x2,y2=batch2


            pred_logits_1=self.forward(x1,invert_mask=True)
            pred_logits_2=self.forward(x2,invert_mask=True)

            acc1s.append(utils.calculate_accuracy(pred_logits_1,y1))
            acc2s.append(utils.calculate_accuracy(pred_logits_2,y2))



        acc1=round(np.mean(acc1s),2)
        acc2=round(np.mean(acc2s),2)

        if self.logging:
            wandb.define_metric("Eval accuracies",step_metric='epoch')
            wandb.log({'Eval accuracies':{"Task":acc1,"NOT task":acc2}},step=self.global_step)
        else:
            print({"Acc task'":acc1,"Acc not task":acc2})


    def MaskedLinear(self,x,name,invert=False):

        '''
        Think invert detaches tensor from comp graph, so should only be used during val
        '''
        binaries=self.transform_logit_tensors() #we could just update binaries every training step
        binary_weight,binary_bias=binaries[name+'.weight'],binaries[name+'.bias']
        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()

            '''
            idxs0_w,idxs1_w=binary_weight==0.0,binary_weight==1.0
            idxs0_b,idxs1_b=binary_bias==0.0,binary_bias==0.0
            binary_weight[idxs0_w]+=1.0
            binary_weight[idxs1_w]-=-1.0
            binary_bias[idxs0_b]=+1.0
            binary_bias[idxs1_b]-=1.0
            '''

        masked_weight,masked_bias=self.param_dict[name+'.weight']*binary_weight,self.param_dict[name+'.bias']*binary_bias
        out=F.linear(x,weight=masked_weight,bias=masked_bias)
        return out

    def MaskedConv2d(self,x,name,bias=False,invert=False):

        '''
        invert detaches tensor from comp graph, so should only be used during val
        '''

        stride,padding=self.leaf_modules[name].stride,self.leaf_modules[name].padding

        binaries=self.transform_logit_tensors()
        binary_weight=binaries[name+'.weight']

        if bias:
            binary_bias=binaries[name+'.bias']
        else:
            masked_bias=None

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            if bias:
                binary_bias=(~(binary_bias.bool())).int()

        masked_weight=self.param_dict[name+'.weight']*binary_weight
        masked_bias=self.param_dict[name+'.bias']*binary_bias
        out=F.conv2d(x,weight=masked_weight,bias=masked_bias,stride=stride,padding=padding)
        return out

    def MaskedBatchNorm2d(self,x,name,invert=False):
        
        #these are approximations to feature mean + variance over whole dataset, calculated during training
        running_mean=self.leaf_modules[name].running_mean
        running_var=self.leaf_modules[name].running_var 

        binaries=self.transform_logit_tensors()
        binary_weight=binaries[name+'.weight']
        binary_bias=binaries[name+'.bias']

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()
        
        masked_weight=self.param_dict[name+'.weight']*binary_weight
        masked_bias=self.param_dict[name+'.bias']*binary_bias
        return F.batch_norm(x,running_mean=running_mean,running_var=running_var,weight=masked_weight,bias=binary_bias)

    def MaskedLayerNorm(self,x,name,invert=False):

        normalized_shape=self.leaf_modules[name].normalized_shape

        binaries=self.transform_logit_tensors()
        binary_weight=binaries[name+'.weight']
        binary_bias=binaries[name+'.bias']

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()
        
        masked_weight=self.param_dict[name+'.weight']*binary_weight
        masked_bias=self.param_dict[name+'.bias']*binary_bias

        return F.layer_norm(normalized_shape=normalized_shape,weight=masked_weight,bias=masked_bias)




    def transform_logit_tensors(self):

        tau=self.tau

        U1 = torch.rand(1, requires_grad=True)
        U2 = torch.rand(1, requires_grad=True)

        samples={}
        for k,v in self.logit_tensors_dict.items():
            samples[k]=torch.sigmoid((v - torch.log(torch.log(U1) / torch.log(U2))) / tau)
            

        binaries_stop={}
        for k,v in samples.items():
            with torch.no_grad():
                binaries_stop[k]=(v>0.5).float()-v
        
        binaries={}
        for k,v in binaries_stop.items():
            binaries[k]=v+samples[k]

        return binaries

    def save(self):

        if not os.path.isdir(self.savedir):
            os.mkdir(self.savedir)

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

        fname=os.path.join(self.savedir,f'checkpoint_step={self.global_step}_epoch={self.train_epoch}')
        with open(fname,'wb') as f:
            pickle.dump(save_dict,f)
            print(f'Checkpoint step={self.global_step}, epoch={self.train_epoch} saved')
 
    def load(self,path):

        with open (path,'rb') as f:
            load_dict=pickle.load(f)

        self.alpha=load_dict.get('alpha')
        self.tau=load_dict.get('tau')
        self.global_step=load_dict.get('global_step')
        self.train_epoch=load_dict.get('train_epoch')
        self.logit_tensors_dict=load_dict.get('logit_tensors_dict')
        self.optimiser=load_dict.get('optimiser')
        self.run_id=load_dict.get('run_id')
    



class MaskedMNISTFFN(AbstractMaskedModel):

    def __init__(self,kwargs):
        super().__init__(**kwargs)

        #none mask trainable layers
        self.layer0=nn.Flatten(start_dim=1,end_dim=-1)

    def forward(self, x, invert_mask=False):
        
        
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


        x=self.MaskedConv2d(x,name='scl.vision.net.0',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='scl.vision.net.1',invert=invert_mask)

        x=self.MaskedConv2d(x,name='scl.vision.net.2',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='scl.vision.net.3',bias=True,invert=invert_mask)

        x=self.MaskedConv2d(x,name='scl.vision.net.4',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='scl.vision.net.5',bias=True,invert=invert_mask)

        x=self.MaskedConv2d(x,name='scl.vision.net.6',bias=True,invert=invert_mask)
        x=self.MaskedBatchNorm2d(x,name='scl.vision.net.7',bias=True,invert=invert_mask)

        x_conv_out=self.MaskedConv2d(x,name='scl.vision.net.8',bias=True,invert=invert_mask)

        x1=self.flatten_layer_vision(x_conv_out)
        x1=self.MaskedLinear(x1,name='scl.vision.net.10',invert=invert_mask)
        x1=self.relu(x1)

        #feedforward residual layer
        x2=self.MaskedLinear(x1,name='scl.vision.net.12.net.0',invert=invert_mask)
        x2=self.MaskedLayerNorm(x2,name='scl.vision.net.12.net.1',invert=invert_mask)
        x2=self.relu(x2)
        x2=self.MaskedLinear(x2,name='scl.vision.net.12.net.3',invert=invert_mask)
        out=x2+x1
        
        return out

    def MaskedAttrNet(self,x,invert_mask=False):
        
        shape,heads=x.shape,self.model.scl.attr_heads
        dim=shape[-1]

        #scattering transform
        x=x.reshape(-1,heads,dim//heads)
        x=self.MaskedLinear(x,name='scl.attr_net.mlp.net.0',invert=invert_mask)
        x=self.relu(x)
        x=self.MaskedLinear(x,name='scl.attr_net.mlp.net.2',invert=invert_mask)

        #feed forward residual net
        x1=self.MaskedLinear(x,name= 'scl.ff_residual.net.0',invert=invert_mask)
        x1=self.MaskedLayerNorm(x1,name= 'scl.ff_residual.net.1',invert=invert_mask)
        x1=self.relu(x1)
        x1=self.MaskedLinear(x1,name= 'scl.ff_residual.net.3',invert=invert_mask)
        out=x+x1


    def MaskedRelNet(self,x,invert_mask=False):

        x=self.MaskedLinear(x,name= 'scl.rel_net.net.0',invert=invert_mask)
        x=self.relu(x)
        x=self.MaskedLinear(x,name= 'scl.rel_net.net.2',invert=invert_mask)
        x=self.relu(x)
        self.MaskedLinear(x,name= 'scl.rel_net.net.4',invert=invert_mask)

        return x

    def MaskedToLogit(self,x,invert_mask=False):

        x=self.MaskedLinear(x,name='scl.to_logit',invert=invert_mask)
        return X





    def forward(self, x, invert_mask=False):

        b,m,n,c,h,w=x.shape
        images=x.view(-1,c,h,w)


        features=self.MaskedVisionNet(x,invert_mask=invert_mask)

        attrs=self.MaskedAttrNet(x,invert_mask=invert_mask)
        attrs=attrs.reshape(b,m,n,self.model.scl.rel_heads,-1).transpose(-2,-3).flatten(3)

        rels=self.MaskedRelNet(x,invert_mask=invert_mask)
        rels=rels.flatten(2)

        logits=self.MaskedToLogit(rels).flatten(1)
    

        return logits
