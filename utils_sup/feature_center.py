import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureCenter(nn.Module):

    def __init__(self, feature_dim, bucket_start=3, bucket_end=96, momentum=0.9):
        super(FeatureCenter, self).__init__()
        self.feature_dim = feature_dim
        self.bucket_end = bucket_end
        self.bucket_start = bucket_start
        self.momentum = momentum

        self.register_buffer('curr_feature_center',torch.zeros((bucket_end-bucket_start,feature_dim)))
        self.register_buffer('last_feature_center',torch.zeros((bucket_end-bucket_start,feature_dim)))

    def reset(self):
        self.curr_feature_center.zero_() 
        self.last_feature_center.zero_()

    def update_last_epoch_stats(self, epoch):
        mask = self.curr_feature_center.sum(dim=1) == 0
        mask = mask.unsqueeze(dim=1)
        mask.expand((self.bucket_end-self.bucket_start,self.feature_dim))
        # Momentum Update  2023-09-26
        self.last_feature_center = self.momentum * self.last_feature_center + \
                                   (1 - self.momentum) * self.curr_feature_center
        #self.last_feature_center = torch.where(mask, self.last_feature_center, self.curr_feature_center)

    def update_running_stats(self, features, labels, epoch):
        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        self.curr_feature_center = torch.nan_to_num(
            torch.vstack([features[labels == i + self.bucket_start].mean(dim=0) for i in range(self.bucket_end - self.bucket_start)])
        )
    
    def init(self, features, labels):
        assert self.feature_dim == features.size(1), "Input feature dimension is not aligned!"
        assert features.size(0) == labels.size(0), "Dimensions of features and labels are not aligned!"

        self.curr_feature_center = torch.nan_to_num(
            torch.vstack([features[labels == i + self.bucket_start].mean(dim=0) for i in range(self.bucket_end - self.bucket_start)])
        )
        
        mask = self.curr_feature_center.sum(dim=1) == 0
        mask = mask.unsqueeze(dim=1)
        mask.expand((self.bucket_end-self.bucket_start,self.feature_dim))
        if not self.last_feature_center.is_cuda:
            self.last_feature_center = self.last_feature_center.cuda()
        self.last_feature_center = torch.where(mask, self.last_feature_center, self.curr_feature_center)

    def get_probability(self, x, T=1.0):
        dist = torch.cdist(x, self.last_feature_center)
        return F.softmax(-dist * T, dim=1)

  
