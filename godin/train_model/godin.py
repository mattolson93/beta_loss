# -*- coding: utf-8 -*-
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Created on Sat Sep 19 20:55:56 2015

@author: liangshiyu
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import math
import argparse
import os

from sklearn.metrics import roc_auc_score, roc_curve
from torch.autograd import Variable

from torch.utils.data import DataLoader, Subset

from densenet import DenseNet3
from deconfnet import DeconfNet, NormalLinear, GodinLayer, RatioLayer, SoftplusLinear, GodinLayerSoftplus
from datasets import OpenDatasets


from tqdm import tqdm

r_mean = 125.3/255
g_mean = 123.0/255
b_mean = 113.9/255
r_std = 63.0/255
g_std = 62.1/255
b_std = 66.7/255

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding = 4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])

test_transform = transforms.Compose([
    transforms.CenterCrop((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
])




class losses:
    def __init__(self, do_softplus=False):
        self.softplus = None#nn.Softplus() if do_softplus else lambda x: x


    def kliep_loss(self, logits, labels, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        

        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def kliep_loss_sigmoid(self, logits, labels, max_ratio=10):
        preds  = torch.sigmoid(logits) * 10

        maxlog = torch.log(torch.FloatTensor([max_ratio])).to(preds.device)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (maxlog-torch.log(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def ulsif_loss(self, logits, labels, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        inlier_loss  = (labels * (-2*(preds))).sum(1)
        outlier_loss = ((1-labels) * (preds**2)).mean(1)
        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss


    def power_loss(self, logits, labels, alpha=.1, max_ratio=50):
        preds = torch.clamp(logits,min=-1*max_ratio, max=max_ratio)
        
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




def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
    # Device arguments
    parser.add_argument('--gpu', default = 0, type = int,
                        help = 'gpu index')

    # Model loading arguments
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--model-dir', default = './outmodels', type = str,
                        help = 'model name for saving')

    # Architecture arguments
    parser.add_argument('--architecture', default = 'densenet', type = str,
                        help = 'underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument('--layer-type', default = 'cosine', type = str,
                        help = 'similarity function for decomposed confidence numerator (cosine | inner | euclid | baseline)')
    parser.add_argument('--loss-type', default = 'ce', type = str,
                        help = 'ce|kliep|power05|power10|power50|power90|ulsif')
    parser.add_argument('--opencat-exp', default = 'None', type = str,
                        help = 'mnist|svhn|cifar10|cifar10+|cifar50+|tinyimagenet')

    # Data loading arguments
    parser.add_argument('--data-dir', default='./data', type = str)
    parser.add_argument('--out-dataset', default = 'Imagenet', type = str,
                        help = 'out-of-distribution dataset')
    parser.add_argument('--batch-size', default = 64, type = int,
                        help = 'batch size')
    parser.add_argument('--do-softplus', default = 1, type = int,
                        help = 'batch size')


    # Training arguments
    parser.add_argument('--no-train', action='store_false', dest='train')
    parser.add_argument('--weight-decay', default = 0.0001, type = float,
                        help = 'weight decay during training')
    parser.add_argument('--epochs', default = 300, type = int,
                        help = 'number of epochs during training')
    parser.add_argument('--seed', default = 0, type = int,
                        help = 'number of epochs during training')

    # Testing arguments
    parser.add_argument('--no-test', action='store_false', dest='test')
    parser.add_argument('--magnitudes', nargs = '+', default = [0, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.005, 0.01, 0.02, 0.04, 0.08], type = float,
                        help = 'perturbation magnitudes')
    
    
    parser.set_defaults(argument=True)
    return parser.parse_args()

    

def get_opencat(data_dir, oc_name, batch_size, seed=1):
    dataObject = OpenDatasets(batch_size, batch_size, num_workers=4, add_noisy_instances=False)

    opencat_dict =  {
        "mnist":        ("mnist","mnist",6,4),
        "svhn":         ("svhn","svhn",6,4),
        "cifar10":      ("cifar10","cifar10",6,4),
        "cifar10+":     ("cifar10","cifar100",4,10),
        "cifar50+":     ("cifar10","cifar100",4,50),
        "tinyimagenet": ("tinyimagenet","tinyimagenet",20,200),
    }

    inlier_dataset, outlier_dataset, inlier_count, outlier_count = opencat_dict[oc_name]
    

    sectioned_inliers, sectioned_outliers = None, None

    if inlier_dataset == 'cifar10' and outlier_dataset  == 'cifar100':
        if inlier_count != 4 or outlier_count <= 0: exit("cannot run cifar10 vs cifar100 unless inlier is 4 and outlier >0")
        sectioned_inliers, sectioned_outliers = dataObject.get_random_cifar100(seed, outlier_count)
    elif inlier_dataset == outlier_dataset:
        print("using only one dataset ")
        num_total_classes = dataObject.get_dataset_class_count(inlier_dataset)
        if inlier_count >= num_total_classes: raise "inlier_count too high"
        sectioned_inliers, sectioned_outliers = dataObject.set_random_classes(seed, inlier_count, num_total_classes)

    elif outlier_count > 0:
        exit("ARGUMENT 'outlier_count' not implemented excpet for cifar10 vs cifar100")

   

    train_loader   = dataObject.get_dataset_byname(inlier_dataset, train=True, specific_classes=sectioned_inliers )  
    test_loader    = dataObject.get_dataset_byname(inlier_dataset, train=False,specific_classes=sectioned_inliers)  
    openset_loader = dataObject.get_dataset_byname(outlier_dataset,train=False,specific_classes=sectioned_outliers)  

    return train_loader, None, test_loader, openset_loader, inlier_count

def get_datasets(data_dir, data_name, batch_size):

    train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=True, download=True, transform=train_transform)
    test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar10', train=False, download=True, transform=test_transform)
    
    if data_name == 'Gaussian' or data_name == 'Uniform':
        normalizer = Normalizer(r_mean, g_mean, b_mean, r_std, g_std, b_std)
        outlier_loader = generating_loaders_dict[data_name](batch_size = batch_size, num_batches = int(10000 / batch_size), transformers = [normalizer])
    else:
        outlier_set  = torchvision.datasets.ImageFolder(f'{data_dir}/{data_name}', transform=test_transform)
        outlier_loader       =  DataLoader(outlier_set,       batch_size=batch_size, shuffle=False, num_workers=4)
    
    test_indices      = list(range(len(test_set_in)))
    validation_set_in = Subset(test_set_in, test_indices[:1000])
    test_set_in       = Subset(test_set_in, test_indices[1000:])

    train_loader_in      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
    validation_loader_in =  DataLoader(validation_set_in, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader_in       =  DataLoader(test_set_in,       batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader_in, validation_loader_in, test_loader_in, outlier_loader

def get_cifar100(data_dir, batch_size):

    train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar100', train=True, download=True, transform=train_transform)
    test_set_in  = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar100', train=False, download=True, transform=test_transform)
    train_loader_in      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
   
    return train_loader_in


def main():
    args = get_args()
    
    device           = "cuda"

    layer_type       = args.layer_type
    loss_type        = args.loss_type
    
    data_dir         = args.data_dir
    model_dir        = args.model_dir
    data_name        = args.out_dataset
    batch_size       = args.batch_size
    
    weight_decay     = args.weight_decay
    epochs           = args.epochs
    seed             = args.seed

    if layer_type == "linear" and loss_type != 'ce': exit("linear layer can only be used with CrossEntropyLoss")
    np.random.seed(seed)
    torch.manual_seed(seed)

    if args.opencat_exp == "cifar100":
        dataset_name= args.opencat_exp
        train_set_in = torchvision.datasets.CIFAR10(root=f'{data_dir}/cifar100', train=True, download=True, transform=train_transform)
        train_data      =  DataLoader(train_set_in,      batch_size=batch_size, shuffle=True,  num_workers=4)
        num_classes = 100

    elif args.opencat_exp != "None":
        dataset_name = args.opencat_exp
        if dataset_name == "mnist": exit("havent implemented mnist yet for densenet")
        train_data, val_data, test_data, open_data, num_classes = get_opencat(data_dir, args.opencat_exp, batch_size, seed)
    else:
        #get outlier data
        dataset_name = 'cifarall'
        num_classes = 10
        train_data, val_data, test_data, open_data = get_datasets(data_dir, data_name, batch_size)

    # Create necessary directories
    exp_name = f'{loss_type}_{layer_type}_{dataset_name}_{seed}'

    print("running experiment of name: ",exp_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    CustomLayers = {
        'linear': NormalLinear,
        'godin': GodinLayer,
        'ratio':RatioLayer,
        'linearsp': SoftplusLinear,
        'godinsp': GodinLayerSoftplus
    }

    underlying_net = DenseNet3(depth = 100, num_classes = num_classes).to(device)
    last_layer = CustomLayers[layer_type](underlying_net.output_size, num_classes).to(device)


    optimizer = optim.SGD(underlying_net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    h_wd = 0 if layer_type == 'godin' else weight_decay 
    h_optimizer = optim.SGD(last_layer.parameters(), lr = 0.1, momentum = 0.9, weight_decay = h_wd) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    model = DeconfNet(underlying_net, last_layer).to(device)


  
    criterion = losses()
    criterion = criterion.get_loss_dict()[loss_type]

   
    model.train()
    
    num_batches = len(train_data)
    epoch_bar = tqdm(total = num_batches * epochs, initial = 0)
    epoch_loss = 0
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_data):
            inputs = inputs.to(device)
            targets = targets.to(device)
            h_optimizer.zero_grad()
            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            
            optimizer.step()
            h_optimizer.step()
            total_loss += loss.item()
            
        epoch_bar.set_description(f'{epoch + 1}/{epochs}|loss{epoch_loss:0.2f}|bat{batch_idx + 1}/{num_batches}')
        epoch_bar.update()

        epoch_loss = total_loss
        h_scheduler.step()
        scheduler.step()
    
    outmodel_pth = os.path.join(model_dir,exp_name+'.pth')
    print("saving model to ", outmodel_pth)    
    torch.save(model.state_dict(), outmodel_pth) # For exporting / sharing / inference only
    
   
if __name__ == '__main__':
    main()
