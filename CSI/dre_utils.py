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
from grad_clamp import dclamp



from tqdm import tqdm

r_mean = 125.3/255
g_mean = 123.0/255
b_mean = 113.9/255
r_std = 63.0/255
g_std = 62.1/255
b_std = 66.7/255

def get_train_transform32():
    return transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
    ])

def get_test_transform32():
    return transforms.Compose([
        transforms.CenterCrop((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
    ])

eps = np.finfo(np.float32).eps 
one_m_eps = 1 - eps
logit_range = -np.log(eps)

class Net(torch.nn.Module):
    def __init__(self, is_ratio):
        super(Net, self).__init__()
        dim = 256
        self.is_ratio = is_ratio
        self.fc1 = nn.Linear(1, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc4 = nn.Linear(dim, 1)
        
    def forward(self, x):
        inshape = x.shape
        x = x.flatten().unsqueeze(1)
        if self.is_ratio:
            x = torch.log(x)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc4(x)
        return torch.exp(x).squeeze().reshape(inshape)

class Losses:
    def __init__(self, do_softplus=True):
        self.softplus = None#nn.Softplus() if do_softplus else lambda x: x
        self.max_clip = torch.tensor(100, dtype=torch.float).cuda()
        self.inv_link = self.identity
        self.loss_model_loaded=False
        self.mlp_model_dir = "./mlp_models"

    def identity(self, x): return x

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

    def set_lossmodel(self,a,b,is_ratio):
        if self.loss_model_loaded: return
        print("a: ", a, "b: ", b, "is_ratio: ", is_ratio)
       
        self.infunc_model = Net(is_ratio).cuda()
        self.infunc_model.load_state_dict(torch.load(os.path.join(self.mlp_model_dir, f"a{a}b{b}_L1.pt")))
        self.outfunc_model = Net(is_ratio).cuda()
        self.outfunc_model.load_state_dict(torch.load(os.path.join(self.mlp_model_dir, f"a{a}b{b}_L0.pt")))
        self.loss_model_loaded=True

    def get_lossmodel(self, logits, labels,  a = -.5, b=-.5, is_ratio=True):
        self.set_lossmodel(a,b,is_ratio)
        probs = logits
        
        y = torch.eye(probs.size(1))
        labels = y[labels].to(probs.device)


        
        use_logit_in  = labels     * (logits <  499.999999).int() if is_ratio else labels     * (logits <  logit_range).int() 
        use_logit_out = (1-labels) * (logits > eps        ).int() if is_ratio else (1-labels) * (logits > -logit_range).int() 

        inlier_loss  = self.infunc_model(probs) * use_logit_in
        outlier_loss = self.outfunc_model(probs) * use_logit_out

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss/(logits.shape[1] - 1)).mean() 
        #loss = (inlier_loss + outlier_loss).mean()
        return loss

    def power_loss_05(self, logits, labels): return self.power_loss(logits, labels, alpha=.05, max_ratio=50)
    def power_loss_10(self, logits, labels): return self.power_loss(logits, labels, alpha=.1, max_ratio=50)
    def power_loss_50(self, logits, labels): return self.power_loss(logits, labels, alpha=.5, max_ratio=50)
    def power_loss_90(self, logits, labels): return self.power_loss(logits, labels, alpha=.90, max_ratio=50)

    def model_a50_b40_ratio(self, logits, labels): return self.get_lossmodel(logits, labels,  a = -.5, b=-.4, is_ratio=True)
    def model_a50_b40_logit(self, logits, labels): return self.get_lossmodel(logits, labels,  a = -.5, b=-.4, is_ratio=False)

    def beta_a50_b50(self, logits, labels): return self.trapz_loss(logits, labels,  a = -.5, b=-.5)
    def beta_a50_b40(self, logits, labels): return self.trapz_loss(logits, labels,  a = -.5, b=-.4)
    def beta_a50_b50sigmoid(self, logits, labels): 
        self.inv_link = torch.sigmoid
        return self.trapz_loss(logits, labels,  a = -.5, b=-.5)

    def beta_a50_b50ratio(self, logits, labels): 
        self.inv_link = lambda x: (x/x+1)
        return self.trapz_loss(logits, labels,  a = -.5, b=-.5)

    def L1_prime(self, q,a,b):
        return (torch.pow(dclamp(q,min=eps),(a-1))) * torch.pow(dclamp(1-q,min=eps),b)
    
    def L0_prime(self, q,a,b):
        return (torch.pow(dclamp(q,min=eps),a))     * torch.pow(dclamp(1-q,min=eps),(b-1))
    
    def trapz_loss(self, logits, labels, a = -.5, b=-.5):
        #import pdb; pdb.set_trace()
        logits = self.inv_link(logits)
        y = torch.eye(logits.size(1))
        labels = y[labels].to(logits.device)
        
        num_tzoids = 3000 if logits.shape[1] < 30 else 350
        
        
        preds = logits.unsqueeze(0).repeat(num_tzoids,1,1)
        
        
        lower_preds = preds * torch.linspace(eps,1.0,num_tzoids).unsqueeze(1).unsqueeze(1).cuda()
        upper_preds = 1- ((1-preds) * torch.linspace(1.0,eps,num_tzoids).unsqueeze(1).unsqueeze(1).cuda())

        
        infunc  = torch.trapz(self.L1_prime(upper_preds,a,b), upper_preds, dim=0)
        outfunc = torch.trapz(self.L0_prime(lower_preds,a,b), lower_preds, dim=0)


        use_logit_in  = labels     * (logits < one_m_eps).int()
        use_logit_out = (1-labels) * (logits > eps).int() 



        inlier_loss  = dclamp(infunc  * use_logit_in , self.max_clip) if self.max_clip > 0 else infunc  * use_logit_in
        outlier_loss = dclamp(outfunc * use_logit_out, self.max_clip) if self.max_clip > 0 else outfunc * use_logit_out

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss + outlier_loss).mean()#/preds.size(1)

        return loss

    def trapzab_softmax(self, logits, labels, a,b):
        probs = torch.softmax(logits,dim=1)
        
        y = torch.eye(probs.size(1))
        labels = y[labels].to(probs.device)
        
        num_tzoids = 5000

        use_logit_in  = labels.bool()
        use_logit_out = (1-labels) 


        preds = probs[use_logit_in].unsqueeze(0).repeat(num_tzoids,1,1)
        upper_preds = 1- ((1-preds) * torch.linspace(1.0,eps,num_tzoids).unsqueeze(1).unsqueeze(1).cuda())
        inlier_loss  = torch.trapz(self.L1_prime(upper_preds,a,b), upper_preds, dim=0)


        #inlier_loss  = infunc * use_logit_in
        #outlier_loss = outfunc * use_logit_out
        inlier_loss  = dclamp(inlier_loss, self.max_clip)

        self.inlier_loss = inlier_loss
        self.outlier_loss = torch.zeros(1)#outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        loss = (inlier_loss ).mean()#+ outlier_loss/(logits.shape[1] - 1)).mean() 
        #loss = (inlier_loss + outlier_loss).mean()
        return loss

    def trapzab_sigmoid(self, logits, labels, a,b):
        logits =  dclamp(logits,min=-logit_range, max=logit_range)
        probs = torch.sigmoid(logits)
        
        y = torch.eye(probs.size(1))
        labels = y[labels].to(probs.device)
        
        num_tzoids = 1500

       
        use_logit_in  = labels     * (logits <  logit_range).int()
        use_logit_out = (1-labels) * (logits > -logit_range).int() 

        preds = probs.unsqueeze(0).repeat(num_tzoids,1,1)
        
        lower_preds = preds * torch.linspace(eps,1.0,num_tzoids).unsqueeze(1).unsqueeze(1).cuda()
        upper_preds = 1- ((1-preds) * torch.linspace(1.0,eps,num_tzoids).unsqueeze(1).unsqueeze(1).cuda())

        
        infunc  = torch.trapz(self.L1_prime(upper_preds,a,b), upper_preds, dim=0)
        outfunc = torch.trapz(self.L0_prime(lower_preds,a,b), lower_preds, dim=0)

       
        inlier_loss  = dclamp(infunc * use_logit_in, self.max_clip)
        outlier_loss = dclamp(outfunc * use_logit_out, self.max_clip)

        self.inlier_loss = inlier_loss
        self.outlier_loss = outlier_loss
        self.used_logits = str(use_logit_in.sum().item()) + "-"  +str(use_logit_out.sum().item())

        return (inlier_loss + outlier_loss).mean()


    def get_loss_dict(self):
        return {
            'ce':nn.CrossEntropyLoss(),
            'kliep':   self.kliep_loss, 
            'ulsif':   self.ulsif_loss, 
            'ab-sigmoid':   self.trapzab_sigmoid, 
            'ab-softmax':   self.trapzab_softmax, 
            'ab-.5':   self.beta_a50_b50, 
            'ab-.54':   self.beta_a50_b40, 
            'model-.54ratio':   self.model_a50_b40_ratio, 
            'model-.54logit':   self.model_a50_b40_logit, 
            'ab-.5sigmoid':   self.beta_a50_b50sigmoid, 
            'ab-.5ratio':   self.beta_a50_b50ratio, 
            'power05': self.power_loss_05, 
            'power10': self.power_loss_10, 
            'power50': self.power_loss_50, 
            'power90': self.power_loss_90, 
        }


def norm(x):
    n = torch.norm(x, p=2, dim=1)
    x = x / (n.expand(1, -1).t() + .0001)
    return x

class GodinLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GodinLayer, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        self.init_weights()

    def get_modes(): return ['latent_norm','logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x):
        denominators = self.g(x)

        x = norm(x)
        w = norm(self.h.weight)
        numerators = (torch.matmul(x,w.T))

        quotients = numerators / denominators

        if self.training:
            return quotients
        else:
            return quotients, numerators, denominators

class GodinLayerSoftplus(nn.Module):
    def __init__(self, in_features, num_classes):
        super(GodinLayerSoftplus, self).__init__()

        self.h = nn.Linear(in_features, num_classes, bias= False)
        self.g = nn.Sequential(
                nn.Linear(in_features, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        self.softplus = nn.Softplus()

        self.init_weights()

    def get_modes(): return ['latent_norm','logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")

    def forward(self, x, clampval=50):
        denominators = self.g(x)

        x = norm(x)
        w = norm(self.h.weight)
        numerators = (torch.matmul(x,w.T))

        quotients = numerators / denominators

        if self.training:
            return self.softplus(dclamp(quotients, min=-clampval, max=clampval)) 
        else:
            return quotients, numerators, denominators

class NormalLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(NormalLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        
    def get_modes(): return ['ce','max','latent_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        self.z = x
        logits = self.h(x)
        return logits

class SoftplusLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SoftplusLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        self.softplus = nn.Softplus()

    def get_modes(): return ['ce','max','latent_norm']
    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x, clampval=50):
        self.z = x
        x = self.h(x)
        if self.training:
            x = dclamp(x, min=-clampval, max=clampval)
        return self.softplus(x)

class SoftplusLinearDre(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SoftplusLinearDre, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        self.softplus = nn.Softplus()

    def get_modes(): return ['ce','max','latent_norm']
    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x, clampval=50):
        self.z = x
        x = self.h(x)
        if self.training:
            x = dclamp(x, min=-clampval, max=clampval)
        r = self.softplus(x)

        return r / (r+1)

class RatioLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioLayer, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.in_featuressqrt = (in_features **.5)
        self.init_weights()
        self.eps =  1.1920929e-07

    def init_weights(self):
        nn.init.kaiming_normal_(self.inlier.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.outlier.weight.data, nonlinearity = "relu")
        self.inlier.bias.data  = torch.zeros(size = self.inlier.bias.size())
        self.outlier.bias.data = torch.zeros(size = self.outlier.bias.size())

    def forward(self, x, max_ratio=15.942385):
        i = self.inlier(x) #/ self.in_featuressqrt
        o = self.outlier(x) #/ self.in_featuressqrt

        if self.training:
            i = dclamp(i, min=-max_ratio, max=max_ratio)
            o = dclamp(o, min=-max_ratio, max=max_ratio)
       

        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ self.eps)
        if self.training: 
            return dclamp(logits, min = 1.1920929e-07, max=15.942385)
        return logits, i, o

class RatioLayerDre(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioLayerDre, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.in_featuressqrt = (in_features **.5)
        self.init_weights()
        self.eps =  1.1920929e-07

    def get_modes(): return ['latent_norm','logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.inlier.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.outlier.weight.data, nonlinearity = "relu")
        self.inlier.bias.data  = torch.zeros(size = self.inlier.bias.size())
        self.outlier.bias.data = torch.zeros(size = self.outlier.bias.size())

    def forward(self, x, max_ratio=15.942385):
        self.z = x
        i = self.inlier(x) #/ self.in_featuressqrt
        o = self.outlier(x) #/ self.in_featuressqrt

        if self.training:
            i = dclamp(i, min=-max_ratio, max=max_ratio)
            o = dclamp(o, min=-max_ratio, max=max_ratio)
       

        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ self.eps)
        if self.training: 
            logits = dclamp(logits, min = 1.1920929e-07, max=15.942385)
        return logits / (logits+1)

class RatioLayer2(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioLayer2, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.in_featuressqrt = (in_features **.5)
        self.init_weights()
        self.eps =  1.1920929e-07



    def init_weights(self):
        nn.init.kaiming_normal_(self.inlier.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.outlier.weight.data, nonlinearity = "relu")
        self.inlier.bias.data  = torch.zeros(size = self.inlier.bias.size())
        self.outlier.bias.data = torch.zeros(size = self.outlier.bias.size())

    def forward(self, x, max_ratio=15.942385):
        self.z = x
        i = self.inlier(x) #/ self.in_featuressqrt
        o = self.outlier(x) #/ self.in_featuressqrt

        if self.training:
            i = dclamp(i, min=-max_ratio, max=max_ratio)
            o = dclamp(o, min=-max_ratio, max=max_ratio)
       

        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ self.eps)
        if self.training: 
            return dclamp(logits, min = 1.1920929e-07, max=500)
        return logits



CustomLayers = {
    'linear': NormalLinear,
    'godin': GodinLayer,
    'ratio':RatioLayer,
    'ratio2':RatioLayer2,
    'dreratio': RatioLayerDre,
    'linearsp': SoftplusLinear,
    'linearspr': SoftplusLinearDre,
    'godinsp':GodinLayerSoftplus,
}

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


def calc_tnr(id_test_results, ood_test_results, tpr_thresh=.95):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=tpr_thresh)]

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    trues  = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result


def get_score(preds, mode):
    if mode  in ['ce','h_ce','g_ce','logit_ce']:# 'confidence_threshold':
        return torch.max(torch.softmax(preds, dim=1), dim=1)[0].data.cpu().numpy()
    elif mode == 'augmented_classifier':
        return preds[:, -1].data.cpu().numpy()
        #return torch.softmax(preds, dim=1)[:, -1].data.cpu().numpy()
    elif mode == 'entropy':
        return ((preds * torch.log(preds)).sum(1)).data.cpu().numpy()
    elif mode  in ['max','h_max','g_max','logit_max']:
        return torch.max(preds, dim=1)[0].data.cpu().numpy()
    elif mode  in ['norm','h_norm','g_norm','logit_norm','latent_norm']:
        #return torch.mean(preds, dim=1).data.cpu().numpy()
        return torch.norm(preds,p=1, dim=1).data.cpu().numpy()
    else:
        print("bad mode of ", mode)
        exit()

@torch.no_grad()
def get_model_preds(model, dataset, modes, get_acc=False):
    device = next(model.parameters()).device
    ret = {}
    for m in modes: ret[m] = []
    total_correct = 0
    total = 0

    for batch in dataset:
        if len(batch) == 2:
            images, labels = batch
            logits = model(images.to(device))
        else:
            input_ids, input_mask, segment_ids, labels = batch
            logits = model(input_ids.to(device), segment_ids.to(device), input_mask.to(device))
        
        if get_acc:
            preds = logits[0] if '_' in modes[0] else logits
            total_correct += torch.sum(preds.max(dim=1)[1].cpu() == labels)
            total+=preds.size(0)



        for m in modes: 
            if 'latent' in m:
                score = get_score(model.fc.z, m)
            elif 'logit_' in m:
                score= get_score(logits[0], m)
            elif 'h_' in m:
                score= get_score(logits[1], m)
            elif 'g_' in m:
                score= get_score(logits[2], m)
            else:
                score= get_score(logits, m)
            ret[m].extend(score)
    
    if get_acc: return ret, (float(total_correct)/total  )  
    return ret



@torch.no_grad()
def save_openset_all(model, testing_dataset, openset_dataset, modes, out_file, exp_id):
    known_scores, acc = get_model_preds(model, testing_dataset, modes, get_acc=True)
    unknown_scores = get_model_preds(model, openset_dataset, modes)

    best_auc = 0
    outstr = ""
    for m in modes:
        k, u = known_scores[m], unknown_scores[m]
        if calc_auroc(k, u) < .5:
            u, k = known_scores[m], unknown_scores[m]

        auc = calc_auroc(k, u) 
        tnr = calc_tnr(k, u) 
        outstr += f',{auc:.4f},{tnr:.4f}'
        if auc > best_auc:
            best_auc = auc
            best_tnr = tnr
            best_mode = m

        print(f'{auc:.4f} AUC, {tnr:.4f} tnr.  Mode {m}') 
        print(f'avg   known:  {np.mean(known_scores[m]):.4f}~ {np.std(known_scores[m]):.4f}')
        print(f'avg unknown:  {np.mean(unknown_scores[m]):.4f}~ {np.std(unknown_scores[m]):.4f}')
    
    best_str = f'{exp_id},{acc:.4f},{best_mode},{best_auc:.4f},{best_tnr:.4f}'
    outstr = best_str + outstr

    with open(out_file, "a") as writer:
        writer.write(outstr+"\n")

    print('id,acc,best_mode,best_auc,best_tnr,',modes)
    print("best,e,   acc, mode, auc, tnr")
    print("best",best_str)
    return best_auc

@torch.no_grad()
def test_open_set_performance2(model, testing_dataset, openset_dataset, modes, get_acc=False):
    model.eval()
    device = next(model.parameters()).device
    known_scores = []
    unknown_scores = []
    total_correct = 0
    total = 0
    known_scores   = get_model_preds(model, testing_dataset, modes)
    unknown_scores = get_model_preds(model, openset_dataset, modes)


    best_auc = 0
    for m in modes:
        auc = calc_auroc(known_scores[m], unknown_scores[m])
        auc = max(auc, 1-auc)
        best_auc = max(auc, best_auc)
        print(f'{auc:.4f} AUC SCORE.  Mode {m}') 
        print(f'avg   known:  {np.mean(known_scores[m]):.4f}~ {np.std(known_scores[m]):.4f}')
        print(f'avg unknown:  {np.mean(unknown_scores[m]):.4f}~ {np.std(unknown_scores[m]):.4f}')
    
    print("best_auc",best_auc)
    return best_auc

@torch.no_grad()
def test_open_set_performance(model, testing_dataset, openset_dataset, mode, get_acc=False):
    model.eval()
    device = next(model.parameters()).device
    known_scores = []
    unknown_scores = []
    total_correct = 0
    total = 0

    for images, labels in testing_dataset:
        logits = model(images.to(device))
        known_scores.extend(get_score(logits, mode))
        if get_acc:
            total_correct += torch.sum(logits.max(dim=1)[1].cpu() == labels)
            total+=logits.size(0)

    for images, labels in openset_dataset:
        logits = model(images.to(device))
        unknown_scores.extend(get_score(logits, mode))

    
    if get_acc: print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, float(total_correct) / total))

    auc = calc_auroc(known_scores, unknown_scores)
    print(f'{auc:.4f} AUC SCORE.  Mode {mode}') 
    print(f'avg   known:  {np.mean(known_scores):.4f}~ {np.std(known_scores):.4f}')
    print(f'avg unknown:  {np.mean(unknown_scores):.4f}~ {np.std(unknown_scores):.4f}')
    return auc

def test_model(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    total = 0
    total_correct = 0
    with torch.no_grad():
        for images, labels in dataset:
            logits = model(images.to(device))



            correct = torch.sum(logits.max(dim=1)[1].cpu() == labels)
            total += len(images)
            total_correct += correct
        accuracy = float(total_correct) / total
        print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
        return accuracy






def testData(model, CUDA_DEVICE, data_loader, noise_magnitude, criterion, score_func = 'h', title = 'Testing', get_acc=False):
    model.eval()
    num_batches = len(data_loader)
    results = []
    data_iter = tqdm(data_loader)

    total = 0
    total_correct = 0

    for j, (images, labels) in enumerate(data_iter):
        data_iter.set_description(f'{title} | Processing image batch {j + 1}/{num_batches}')
        images = Variable(images.to(CUDA_DEVICE), requires_grad = True)
        
        
        logits, h, g = model(images)

        if score_func == 'h':
            scores = h
        elif score_func == 'g':
            scores = g
        elif score_func == 'logit':
            scores = logits

        if get_acc:
            total += len(images)
            total_correct += torch.sum(logits.max(dim=1)[1].cpu() == labels)

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of the numerator w.r.t. input
        max_scores, _ = torch.max(scores, dim = 1)
        max_scores.backward(torch.ones(len(max_scores)).to(CUDA_DEVICE))
        
        # Normalizing the gradient to binary in {-1, 1}
        if images.grad is not None:
            gradient = torch.ge(images.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            # Normalizing the gradient to the same space of image
            gradient[::, 0] = (gradient[::, 0] )/(63.0/255.0)
            gradient[::, 1] = (gradient[::, 1] )/(62.1/255.0)
            gradient[::, 2] = (gradient[::, 2] )/(66.7/255.0)
            # Adding small perturbations to images
            tempInputs = torch.add(images.data, gradient, alpha=noise_magnitude)
        
            # Now calculate score
            logits, h, g = model(tempInputs)

            if score_func == 'h':
                scores = h
            elif score_func == 'g':
                scores = g
            elif score_func == 'logit':
                scores = logits

        results.extend(torch.max(scores, dim=1)[0].data.cpu().numpy())
        
    data_iter.set_description(f'{title} | Processing image batch {num_batches}/{num_batches}')
    data_iter.close()

    if get_acc:
        accuracy = float(total_correct) / total
        print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))


    return np.array(results)

if __name__ == '__main__':
    main()
