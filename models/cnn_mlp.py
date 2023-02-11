import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import pytorch_lightning as pl
import importlib
import data

from torch.nn import functional as F




#models
class conv_module(nn.Module):
    def __init__(self):
        super(conv_module, self).__init__()
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        self.batch_norm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(self.batch_norm1(x))
        x = self.conv2(x)
        x = self.relu2(self.batch_norm2(x))
        x = self.conv3(x)
        x = self.relu3(self.batch_norm3(x))
        x = self.conv4(x)
        x = self.relu4(self.batch_norm4(x))
        return x.view(x.size()[0],-1) #feature summaries for all elements in batch


class mlp_module(nn.Module):
    def __init__(self):
        super(mlp_module, self).__init__()
        self.fc1 = nn.Linear(32*4*4, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class FCTreeNet(torch.nn.Module):
    def __init__(self, in_dim=300, img_dim=256, use_cuda=True):
        '''
        initialization for TreeNet model, basically a ChildSumLSTM model
        with non-linear activation embedding for different nodes in the AoG.
        Shared weigths for all LSTM cells.
        :param in_dim:      input feature dimension for word embedding (from string to vector space)
        :param img_dim:     dimension of the input image feature, should be (panel_pair_number * img_feature_dim (e.g. 512 or 256))
        '''
        super(FCTreeNet, self).__init__()
        self.in_dim = in_dim
        self.img_dim = img_dim
        self.fc = nn.Linear(self.in_dim, self.in_dim)
        self.leaf = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.middle = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.merge = nn.Linear(self.in_dim + self.img_dim, self.img_dim)
        self.root = nn.Linear(self.in_dim + self.img_dim, self.img_dim)

        self.relu = nn.ReLU()

    def forward(self, image_feature, input, indicator):
        '''
        Forward funciton for TreeNet model
        :param input:		input should be (batch_size * 6 * input_word_embedding_dimension), got from the embedding vector
        :param indicator:	indicating whether the input is of structure with branches (batch_size * 1)
        :param image_feature:   input dictionary for each node, primarily feature, for example (batch_size * 16 (panel_pair_number) * feature_dim (output from CNN))
        :return:
        '''
        # image_feature = image_feature.view(-1, 16, image_feature.size(2))
        input = self.fc(input.view(-1, input.size(-1)))
        input = input.view(-1, 6, input.size(-1))
        input = input.unsqueeze(1).repeat(1, image_feature.size(1), 1, 1)
        indicator = indicator.unsqueeze(1).repeat(1, image_feature.size(1), 1).view(-1, 1)

        leaf_left = input[:, :, 3, :].view(-1, input.size(-1))           # (batch_size * panel_pair_num) * input_word_embedding_dimension
        leaf_right = input[:, :, 5, :].view(-1, input.size(-1))
        inter_left = input[:, :, 2, :].view(-1, input.size(-1))
        inter_right = input[:, :, 4, :].view(-1, input.size(-1))
        merge = input[:, :, 1, :].view(-1, input.size(-1))
        root = input[:, :, 0, :].view(-1, input.size(-1))
        
        # concating image_feature and word_embeddings for leaf node inputs
        leaf_left = torch.cat((leaf_left, image_feature.view(-1, image_feature.size(-1))), dim=-1)
        leaf_right = torch.cat((leaf_right, image_feature.view(-1, image_feature.size(-1))), dim=-1)

        out_leaf_left = self.leaf(leaf_left)
        out_leaf_right = self.leaf(leaf_right)

        out_leaf_left = self.relu(out_leaf_left)
        out_leaf_right = self.relu(out_leaf_right)

        out_left = self.middle(torch.cat((inter_left, out_leaf_left), dim=-1))
        out_right = self.middle(torch.cat((inter_right, out_leaf_right), dim=-1))

        out_left = self.relu(out_left)
        out_right = self.relu(out_right)

        out_right = torch.mul(out_right, indicator)
        merge_input = torch.cat((merge, out_left + out_right), dim=-1)
        out_merge = self.merge(merge_input)

        out_merge = self.relu(out_merge)

        out_root = self.root(torch.cat((root, out_merge), dim=-1))
        out_root = self.relu(out_root)
        # size ((batch_size * panel_pair) * feature_dim)
        return out_root



class CNN_MLP(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.conv=conv_module()
        self.l1=nn.Linear(2592,32)
        self.l2=nn.Linear(32,8)

    def forward(self,x):
        x=self.conv(x)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        return x

    def training_step(self,batch,idx):
        image,target,meta_target,meta_structure=batch
        logits=self.forward(image)
        loss=F.cross_entropy(logits,target) #compare log probs with target
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters())



#simple model to solve IRAVEN