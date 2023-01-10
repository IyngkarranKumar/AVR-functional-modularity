
#%%
import os
import torch
import torch.nn.functional as F
import pytorch_lightning as pl;
import importlib
import numpy as np
import data

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from copy import deepcopy
from torch.special import logit
importlib.reload(data)

pl.seed_everything(42)

#%% utils 
if 1: 
    def Gumbel_Sigmoid(tens,T=1):
        log_U1=torch.log(torch.rand_like(tens))
        log_U2=torch.log(torch.rand_like(tens))
        t1=(tens-torch.log(log_U1/log_U2))/T
        t2=torch.sigmoid(t1)
        return t2

    def indicator(tens,threshold=0.5,below=0,above=1):

        t1=-1*F.threshold(tens,threshold=threshold,value=below)
        t2=F.threshold(t1,threshold=-0.00001,value=above).int()
        return t2

    
#%% datasets - return train loaders

if 1: 
    batch_size=64
    epochs=2
    subset_frac=0.2


    data_path='datasets'

    dataset = MNIST(data_path, download=True, transform=transforms.ToTensor())
    idxs=np.random.choice(range(dataset.__len__()),int(dataset.__len__()*subset_frac),replace=False)
    subset=torch.utils.data.Subset(dataset,idxs)
    train_loader = DataLoader(subset,batch_size=batch_size,shuffle=True)

    custom_dataset=data.MNISTCustomDataset(n=5)
    custom_train_loader=DataLoader(custom_dataset)
    

#%%toy network training - return trained simple model

if 1: 

    class SimpleModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.l1=nn.Linear(28*28,10)


        def forward(self,x):
            y=self.l1(x)
            return y

    epochs=3
    model=SimpleModel()
    optimiser=torch.optim.Adam(model.parameters())
    criterion=torch.nn.CrossEntropyLoss()

    print('Training simple model')
    for _ in range(epochs):
        running_loss=0
        for i,batch in enumerate(train_loader):
            optimiser.zero_grad()
            x,y=batch
            x=x.squeeze().view(-1,784)
            logits=model(x)
            loss=criterion(logits,y) #pytorch makes it super convenient to use cross entropy
            running_loss+=loss
            loss.backward()
            optimiser.step()
        running_loss/=i #mean loss over epoch
        print(f"Average epoch loss: {running_loss}")

    print('Finished training')

model_pl=list(model.parameters())


#%%
class Indicator(nn.Module):

    def __init__(self,threshold=0.5,below=0,above=1):

        super().__init__()
        self.l1=torch.nn.Threshold(threshold=threshold,value=below)
        self.l2=torch.nn.Threshold(threshold=-1e-10,value=above)

    def forward(self,x):

        x=self.l2(-1*self.l1(x))
        x=x.int()
        return x


class MaskedModel(nn.Module):

    def __init__(self,model):
        
        super().__init__()
        self.masked_model=deepcopy(model)
        self.indicator=Indicator()
        self.trained_weights=list(model.parameters())
        for w in self.trained_weights: w.requires_grad=False #model params are frozen
        self.logit_mask=[nn.Parameter(torch.rand_like(w,requires_grad=True)) for w in self.trained_weights]
        self.binarised_mask=[self.indicator(tens) for tens in self.logit_mask]



        for i,param in enumerate(self.masked_model.parameters()):
            param.data=self.trained_weights[i]*self.binarised_mask[i] #change weights of model


    def forward(self,x):

        logits=self.masked_model(x)
        return logits
        

    def logit_l2_loss(self,mode='mean'):
        l2=0
        for tens in self.logit_mask:
            if mode=='mean':
                l2+=(tens**2).mean()
            elif mode=='sum':
                l2+=(tens**2).sum()
            else:
                raise Exception(f'{mode} is an invalid l2 mode')

        return l2

    def mask_sparsity(self):
        binarised_mask=self.binarised_mask
        binarised_mask=[self.indicator(tens) for tens in self.logit_mask]
        numel=sum(torch.numel(btens) for btens in binarised_mask)
        num_ones=sum([torch.count_nonzero(btens).item() for btens in binarised_mask])
        return ((num_ones/numel)*100)



maskedmodel=MaskedModel(model)


epochs=3
criterion=torch.nn.CrossEntropyLoss()
optimiser=torch.optim.Adam(maskedmodel.parameters())
optimiser.add_param_group({'params':maskedmodel.logit_mask})

print('Training masked model')
for _ in range(epochs):
    running_loss=0
    for i,batch in enumerate(train_loader):
        optimiser.zero_grad()
        x,y=batch
        x=x.squeeze().view(-1,784)
        logits=maskedmodel(x)

        sum1=sum(t1.sum() for t1 in maskedmodel.logit_mask)

        l2_loss=maskedmodel.logit_l2_loss()
        CE_loss=criterion(logits,y)
        loss=CE_loss+l2_loss; running_loss+=loss
        running_loss+=loss

        
        loss.backward()
        optimiser.step()

        sum2=sum(t1.sum() for t1 in maskedmodel.logit_mask)

        #print(f"Delta logit mask {np.abs((sum1-sum2).item())}")
        
    running_loss/=i #mean loss over epoch
    print(f"\n Average epoch loss: {running_loss}")
    print(f'Non-zero proportion {maskedmodel.mask_sparsity()}')

print('\n Finished training')

#%% AE training

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)

class LitAutoEncoder(nn.Module):
    def __init__(self, encoder=Encoder(), decoder=Decoder()):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat


model=LitAutoEncoder(Encoder(),Decoder())
criterion=torch.nn.MSELoss()
optimiser=torch.optim.Adam(model.parameters())

for _ in range(epochs):

    running_loss=0
    for i,batch in enumerate(train_loader):
        optimiser.zero_grad() #reset and calculated gradients
        x,_=batch
        x=x.squeeze().view(-1,784)
        out=model(x)
        loss=criterion(x,out)
        loss.backward()
        optimiser.step()
    running_loss/=i
    print(f"Average epoch loss: {running_loss}")

print('Finished training')


#we also want to get a dataset of JUST ONES, or JUST 10s

#----------------------------TRAIN MODEL-----------------------------------------------





#--------------------------------------------------------------------------------------
