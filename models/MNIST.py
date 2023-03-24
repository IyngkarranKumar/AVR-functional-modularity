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
        self.log('train-loss',loss.item())
        self.log('train-acc',acc)


        return loss

    def validation_step(self,batch,batch_idx,on_epoch=True):
        X,y=batch
        y_hat=self(X)
        loss=F.cross_entropy(y_hat,y)
        probs=F.softmax(y_hat,dim=1)
        pred=torch.argmax(probs,axis=1)
        acc=(len(torch.nonzero(pred==y))/len(pred))*100
        
        #logging
        self.log('val-loss',loss.item())
        self.log('val-acc',acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=0.02)

