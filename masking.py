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

class TestParallelModel(nn.Module):

    def __init__(self):
        super().__init__()

        #branch 1
        self.b1_l1=nn.Linear(in_features=10,out_features=20)
        self.b1_l2=nn.Linear(in_features=20,out_features=40)

        #branch 2
        self.b2_l1=nn.Linear(in_features=10,out_features=30)
        self.b2_l2=nn.Linear(in_features=30,out_features=5)

    def forward(self,x):

        #branch 1 forward
        b1_o1=self.b1_l1(x)
        b1_o2=self.b1_l2(b1_o1)

        #branch 2 forward
        b2_o1=self.b2_l1(x)
        b2_o2=self.b2_l2(b2_o1)

        return b1_o2,b2_o2

model=TestParallelModel()

x=torch.rand(8,10)
o1,o2=model(x)




class AbstractMaskedModel(ABC):

    def __init__(self,model,train_dataloader,test_dataloader1,test_dataloader2,tau):
        
        self.model=model
        self.train_dataloader=train_dataloader
        self.test_dataloader1=test_dataloader1
        self.test_dataloader2=test_dataloader2

        self.logit_tensors_dict={k:torch.nn.Parameter(data=torch.full_like(p,0.9)) for k,p in model.named_parameters()}
        self.alpha=None
        self.tau=tau
        self.logging=False

        self.train_epoch=0


        #freeze model parameters
        for p in model.parameters():
            p.requires_grad=False
        self.param_dict=dict(model.named_parameters())

        self.leaf_modules=get_named_children(self.model)


        self.optimiser=torch.optim.Adam(self.logit_tensors_dict.values())

    @abstractmethod
    def forward(self,x,invert_mask=False):
        pass

    def calculate_loss(self,y_hat,y):
        crossent_loss=F.cross_entropy(y_hat,y)
        reg_loss=self.alpha*torch.sum(torch.stack([torch.sum(logit_tens) for logit_tens in list(self.logit_tensors_dict.values())]))
        loss=crossent_loss+reg_loss
        acc=utils.calculate_accuracy(y_hat,y)

        return crossent_loss,reg_loss,loss,acc


    def train(self,alpha,n_batches=10,n_epochs=5,logging=False,
            val_every_n_steps=10,
            eval_every=10,n_eval_batches=5,norm_freq=5,set_log_name=False):

            #set class attributes for use in rest of class
            self.alpha=alpha
            
            if logging:
                self.logging=True
            if self.logging:
                if set_log_name:
                    log_name=str(input('Enter log name'))
                else:
                    log_name=None
                run=wandb.init(project='AVR',name=log_name)

            for epoch in range(n_epochs):
                for batch_idx,batch in enumerate(self.train_dataloader):
                    if n_batches=='full':
                        pass
                    if batch_idx==n_batches:
                        break
                    x,y=batch
                    y_hat=self.forward(x)

                    crossent_loss,reg_loss,loss,acc=self.calculate_loss(y_hat,y)
                    loss.backward()
                    self.optimiser.step()

                    if self.logging:
                        wandb.log({'epoch':epoch,
                                    'train_loss':loss.item(),
                                    'train_crossent_loss':crossent_loss.item(),
                                    'train_reg_loss':reg_loss.item(),
                                    'train_accuracy':acc
                                    })
                    if (run.step%val_every_n_steps==0) and (run.step!=0):
                        self.validation()

    def validation(self):
        batch=next(iter(self.test_dataloader1))
        x,y=batch
        with torch.no_grad():
            y_hat=self.forward(x)
        crossent_loss,reg_loss,loss,acc=self.calculate_loss(y_hat,y)

        if self.logging:
                wandb.log({
                            'validation_loss':loss.item(),
                            'validation_crossent_loss':crossent_loss.item(),
                            'validation_reg_loss':reg_loss.item(),
                            'validation_accuracy':acc
                            })

    def eval(self,task_eval_dataloader,_task_eval_dataloader,n_batches):
        '''
        Evaluated mask via ablation

        Ablation - frozen_parameters * ~binaries (inverted mask)
        '''
        print('start eval')
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

            pred_logits_1=self.forward(x1,invert=True)
            pred_logits_2=self.forward(x2,invert=True)

            acc1s.append(utils.calculate_accuracy(pred_logits_1,y1))
            acc2s.append(utils.calculate_accuracy(pred_logits_2,y2))

        acc1=round(np.mean(acc1s),2)
        acc2=round(np.mean(acc2s),2)

        if self.logging:
            wandb.define_metric("Eval accuracies",step_metric='epoch')
            wandb.log({'Eval accuracies':{"Task":acc1,"NOT task":acc2}})
        else:
            print({"Acc task'":acc1,"Acc not task":acc2})

        print('end eval')
        


    def MaskedLinear(self,x,name,invert=False):

        '''
        Think invert detaches tensor from comp graph, so should only be used during val
        '''
        binaries=self.transform_logit_tensors() #we could just update binaries every training step
        binary_weight,binary_bias=binaries[name+'.weight'],binaries[name+'.bias']
        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            binary_bias=(~(binary_bias.bool())).int()

        masked_weight,masked_bias=self.param_dict[name+'.weight']*binary_weight,self.param_dict[name+'.bias']*binary_bias
        out=F.linear(x,weight=masked_weight,bias=masked_bias)
        return out

    def MaskedConv2d(self,x,name,bias=False,invert=False):

        '''
        Think invert detaches tensor from comp graph, so should only be used during val
        '''

        binaries=self.transform_logit_tensors()
        binary_weight=binaries[name+'.weight']
        masked_weight=self.param_dict[name+'.weight']*binary_weight

        if bias:
            binary_bias=binaries[name+'.bias']
            masked_bias=self.param_dict[name+'bias']*binary_bias
        else:
            masked_bias=None

        if invert:
            binary_weight=(~(binary_weight.bool())).int()
            if not bias:
                binary_bias=(~(binary_bias.bool())).int()

        out=F.conv2d(x,weight=masked_weight,bias=masked_bias)
        return out


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




class MNISTFFN(pl.LightningModule):

    def __init__(self):
        super().__init__()

        self.layers=nn.Sequential(
            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(28*28,256),
            nn.Linear(256,128),
            nn.Linear(128,64),
            nn.Linear(64,10),
        )
        


    def forward(self,x):
        x=self.layers(x)
        return x

    def training_step(self,batch,batch_idx):
        x,y=batch
        y_hat=self(x)
        loss=F.cross_entropy(y_hat,y)
        probs=F.softmax(y_hat,dim=1)
        pred=torch.argmax(probs,axis=1)
        acc=(len(torch.nonzero(pred==y))/len(pred))*100

        #logging
        self.logger.experiment.add_scalars('Pretrained loss',{'train':loss.item()},self.global_step)
        self.logger.experiment.add_scalars('Pretrained accuracy',{'train':acc},self.global_step)

        return loss

    def validation_step(self,batch,batch_idx,on_epoch=True):
        X,y=batch
        y_hat=self(X)
        loss=F.cross_entropy(y_hat,y)
        probs=F.softmax(y_hat,dim=1)
        pred=torch.argmax(probs,axis=1)
        acc=(len(torch.nonzero(pred==y))/len(pred))*100
        
        #logging
        self.logger.experiment.add_scalars('Pretrained loss',{'val':loss.item()},self.global_step)
        self.logger.experiment.add_scalars('Pretrained accuracy',{'val':acc},self.global_step)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=0.02)

model=MNISTFFN().load_from_checkpoint('logs/lightning_logs/version_7/checkpoints/epoch=4-step=4690.ckpt')

class MaskedMNISTFFN(AbstractMaskedModel):

    def __init__(self,model):
        super().__init__()

        self.layer0=nn.Flatten(start_dim=1,end_dim=-1)

    def forward(self, x, invert_mask=False):
        
        
        x0=self.layer0(x)
        x1=self.MaskedLinear(x0,name='layer1')
        x2=self.MaskedLinear(x1,name='layer2')
        x3=self.MaskedLinear(x2,name='layer3')
        x4=self.MaskedLinear(x3,name='layer4')

        return x4


mm1=MaskedMNISTFFN(model)
