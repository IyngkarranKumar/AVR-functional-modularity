#dataset and dataloader class implementation
#%%

from torch.utils.data import Dataset,DataLoader

import os
import glob
import numpy as np
import torch

from pathlib import Path
from torchvision.transforms import ToTensor

#%%

class RPMdataset(Dataset):

    def __init__(self,root_dir,n,img_size=(160,160),
    transform=ToTensor()):
        super().__init__()
        self.root_dir = root_dir
        self.n = n #number of files in each child dir
        self.files = sorted(list(Path(root_dir).glob("*/*.npz")))
        self.transform = transform


    def __len__(self):
        return len(self.files)



    def __getitem__(self,idx):

        path = self.files[idx]
        data = np.load(path)

        target = data["target"]
        predict = data["predict"]
        image = data["image"]
        meta_matrix = data["meta_matrix"]
        meta_structure = data["meta_structure"]
        meta_target = data["meta_target"]
        structure = data["structure"]

        if self.transform:
            image = self.transform(image)
            target = torch.tensor(target,dtype=torch.long)
            meta_target = self.transform(meta_target)


        return target,predict,image,meta_matrix,meta_structure,meta_target,structure

#%%

dataset = RPMdataset("triangles",n=3)
train_dataloader = DataLoader(dataset,batch_size=3,shuffle=True)