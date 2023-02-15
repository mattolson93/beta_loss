# Custom loss functions
import torch.nn as nn
import time
import os
import torch
import random
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as T

from vector import make_noise
from dataloader import FlexibleCustomDataloader
import imutil
from logutil import TimeSeries

#import pdb; pdb.set_trace()

class losses:
    def __init__(self, do_softplus=True):
        self.softplus = nn.Softplus() if do_softplus else lambda x: x
   
    def kliep_loss(self, logits, labels, max_ratio=50): # max ratio 50 by default
        # We want labels that are NOT one-hot, so we check
        # and correct if this is the case
        #if len(labels.shape) == 2:
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        #preds  = torch.softmax(logits,dim=1)
        preds  = self.softplus(logits)
        #preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def kliep_loss_sigmoid(self, logits, labels, max_ratio=10):
        return -1 #dont use me
        preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def ulsif_loss(self, logits, labels, max_ratio=50):
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        #preds  = torch.softmax(logits,dim=1)
        preds  = self.softplus(logits)
        #preds  = torch.sigmoid(logits) * 10
        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (-2*(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds**2)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def power_loss(self, logits, labels, alpha=.1, max_ratio=50):
        logits = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        preds  = self.softplus(logits)
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)
        inlier_loss  = (labels * (1 - preds.pow(alpha))/(alpha)).sum(1)
        outlier_loss = ((1-labels) * (preds.pow(1+alpha)-1)/(1+alpha)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)
        return loss
    
    def power_loss_05(self, logits, labels): return self.power_loss(logits, labels, alpha=.05, max_ratio=50)
    
    def power_loss_10(self, logits, labels): return self.power_loss(logits, labels, alpha=.1, max_ratio=50)
    
    def power_loss_50(self, logits, labels): return self.power_loss(logits, labels, alpha=.5, max_ratio=50)
    
    def power_loss_90(self, logits, labels): return self.power_loss(logits, labels, alpha=.90, max_ratio=50)
    
    def get_loss_dict(self):
        return {
            'ce':nn.CrossEntropyLoss(),
            'kliep':   self.kliep_loss, 
            'ulsif':   self.ulsif_loss, 
            'power05': self.power_loss_05, 
            'power10': self.power_loss_10, 
            'power50': self.power_loss_50, 
            'power90': self.power_loss_90, 
        }
