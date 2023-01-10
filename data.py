
import numpy as np
import torch
import pytorch_lightning as pl
import zipfile
import os
import idx2numpy

from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
from glob import glob
from skimage.transform import resize
from scipy import misc

#RAVEN module doesn't currently work
class RD(Dataset):

    def __init__(self,root_dir='datasets/RAVEN-10000/center_single',mode='train',transform=None,shuffle=True,img_size=224):
        
        self.root_dir=root_dir
        self.mode=mode
        self.transform=transform
        self.shuffle=shuffle
        self.embeddings = np.load(os.path.join(root_dir, 'embedding.npy'), allow_pickle=True)
        self.img_size=img_size

        glob_wildcard=self.root_dir+'/*'+self.mode+'.npz'
        self.filenames=glob(glob_wildcard)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        data_path=self.filenames[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160)
        target = data["target"]
        structure = data["structure"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(list(indices))
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(resize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)
        # image = resize(image, (16, 128, 128))
        # meta_matrix = data["mata_matrix"]


        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        for element in structure:
            if element != '/':
                embedding[element_idx, :] = torch.tensor(self.embeddings.item().get(element), dtype=torch.float)
                element_idx += 1
        if element_idx == 6:
            indicator[0] = 1.



        # if meta_target.dtype == np.int8:
        #     meta_target = meta_target.astype(np.uint8)
        # if meta_structure.dtype == np.int8:
        #     meta_structure = meta_structure.astype(np.uint8)
    
        del data
        if self.transform:
            resize_image = self.transform(resize_image)
            # meta_matrix = self.transform(meta_matrix)
            target = torch.tensor(target, dtype=torch.long)
            meta_target=meta_target[:,None] #size agreement
            meta_structure=meta_structure[:,None] #size agreement
            meta_target = self.transform(meta_target)
            meta_structure = self.transform(meta_structure)
            # meta_target = torch.tensor(meta_target, dtype=torch.long)
        return resize_image, target, meta_target, meta_structure

class RAVENDataModule(pl.LightningDataModule):

    def __init__(self,batch_size=64):
        self.batch_size=batch_size

    def prepare_data()->None:
        #assume data already downloaded
        pass

    def setup(self,root_dir='datasets/RAVEN-10000',transform=transforms.Compose([transforms.ToTensor()])):


        self.RAVEN_train=RAVENDataset(root_dir=root_dir,dataset_type='train',transform=transform)
        self.RAVEN_val=RAVENDataset(root_dir=root_dir,dataset_type='val',transform=transform)
        self.RAVEN_test=RAVENDataset(root_dir,dataset_type='test',transform=transform)

    def train_dataloader(self):
        RAVEN_train=DataLoader(self.RAVEN_train,batch_size=self.batch_size)
        return RAVEN_train

    def val_dataloader(self):
        RAVEN_val=DataLoader(self.RAVEN_val,batch_size=self.batch_size)
        return RAVEN_val

    def test_dataloader(self):
        RAVEN_test=DataLoader(self.RAVEN_test,batch_size=self.batch_size)
        return RAVEN_test

class IRAVENDataset(Dataset):

    def __init__(self,root='datasets/originals',mode='train',transform=transforms.ToTensor(),shuffle=True,img_size=None):

        self.dataset_dir=os.path.join(root,mode)
        self.mode=mode
        self.transform=transform
        self.shuffle=shuffle
        self.img_size=img_size #resize not implemented yet

        self.filenames=[]
        for subdir in os.listdir(self.dataset_dir):
            glob_wildcard=os.path.join(self.dataset_dir,subdir)+'/*.npz'
            self.filenames=self.filenames+glob(glob_wildcard)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self,idx):
        data_path=self.filenames[idx]
        data = np.load(data_path)
        image = data["image"].reshape(16, 160, 160) #image. duh
        target = data["target"] #correct answer in answer set
        structure = data["structure"] #structure as given by ASIG grammar
        meta_target = data["meta_target"] #unsure
        meta_structure = data["meta_structure"] #unsure

        '''
        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(list(indices))
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(resize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)
        '''
    
        image=transforms.ToTensor()(image).permute(1,0,2)
        target = torch.tensor(target, dtype=torch.long)
        meta_target=meta_target[:,None] #size agreement
        meta_structure=meta_structure[:,None] #size agreement
        meta_target = torch.tensor(meta_target)
        meta_structure = torch.tensor(meta_structure)
        # meta_target = torch.tensor(meta_target, dtype=torch.long)


        return image, target, meta_target, meta_structure
        

class IRAVENDataModule(pl.LightningDataModule):
    
    def __init__(self,batch_size=8):
        self.batch_size=batch_size

    def prepare_data(self)->None:
        #assume data already downloaded
        pass

    def setup(self,root_dir='datasets/originals',transform=transforms.Compose([transforms.ToTensor()])):


        self.IRAVEN_train=IRAVENDataset(root=root_dir,mode='train',transform=transform)
        self.IRAVEN_val=IRAVENDataset(root=root_dir,mode='val',transform=transform)
        self.IRAVEN_test=IRAVENDataset(root=root_dir,mode='test',transform=transform)

    def train_dataloader(self):
        IRAVEN_train=DataLoader(self.IRAVEN_train,batch_size=self.batch_size)
        return IRAVEN_train

    def val_dataloader(self):
        IRAVEN_val=DataLoader(self.IRAVEN_val,batch_size=self.batch_size)
        return IRAVEN_val

    def test_dataloader(self):
        IRAVEN_test=DataLoader(self.IRAVEN_test,batch_size=self.batch_size)
        return IRAVEN_test

class MNISTCustomDataset(Dataset):

    def __init__(self,mode='train',n=0):

        if mode=='train':
            self.img_path='datasets/MNIST/raw/train-images-idx3-ubyte'
            self.label_path='datasets/MNIST/raw/train-labels-idx1-ubyte'
        elif mode=='test':
            self.img_path='datasets/MNIST/raw/t10k-images-idx3-ubyte'
            self.label_path='datasets/MNIST/raw/t10k-labels-idx1-ubyte'

        assert n>=0 and n<=9
        self.n=n
        label_arr=idx2numpy.convert_from_file(self.label_path)
        self.img_file_idxs=np.argwhere(label_arr==n)
        

    def __len__(self):
        return len(self.img_file_idxs)

    def __getitem__(self,idx):
        img_arr=idx2numpy.convert_from_file(self.img_path)[self.img_file_idxs][idx]
        return img_arr,self.n
        
