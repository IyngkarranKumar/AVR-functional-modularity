# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from blocks_2 import *


class CoPINet(nn.Module):

    def __init__(self, num_attr=10, num_rule=6, sample=False, dropout=False):
        super(CoPINet, self).__init__()
        self.num_attr = num_attr
        self.num_rule = num_rule
        self.sample = sample
        
        self.inference_feature_extraction = inference_feature_extraction()
        
        self.predict_rule = nn.Linear(64, self.num_attr * self.num_rule)
        if self.sample:
            self.inference = GumbelSoftmax(temperature=0.5)
        else:
            self.inference = nn.Softmax(dim=-1)
            
        self.inference_rule_transform = inference_rule_transform(num_rule = self.num_rule,num_attr=self.num_attr)
            
        self.perception_feature_extraction = perception_feature_extraction()
        self.contrast = contrast()


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp = MLP(in_dim=256, out_dim=1, dropout=dropout)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, (nn.InstanceNorm2d, nn.BatchNorm2d)):
                if m.affine:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(-1, 16, 80, 80)
        N, _, H, W = x.shape

        
        # Inference Branch
        
        inference_input_features = self.inference_feature_extraction(x)

        predict_rules = self.predict_rule(
            inference_input_features)  # N, self.num_attr * self.num_rule
        predict_rules = predict_rules.view(-1, self.num_rule)
        predict_rules = self.inference(predict_rules) #sample rule
        
        contrast1_bias,contrast2_bias = self.inference_rule_transform(predict_rules)
        

        
        # Perception Branch

        perception_input_features = self.perception_feature_extraction(x)


        contrast_out = self.contrast(perception_input_features,contrast1_bias,contrast2_bias)
        
        
        ##------------Project features to energy space(?)

        avgpool = self.avgpool(contrast_out)
        avgpool = avgpool.view(-1, 256)
        final = avgpool
        final = self.mlp(final)
        
        
        return final.view(-1, 8)