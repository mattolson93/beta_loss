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
from dclamp import clamp


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




class Losses:
    def __init__(self, do_softplus=True):
        self.softplus = nn.Softplus() if do_softplus else lambda x: x
        self.max_clip = torch.tensor(50, dtype=torch.float).cuda()


    def kliep_loss(self, logits, labels, max_ratio=50):
        logits = clamp(logits,min=-1*max_ratio, max=max_ratio)
        
        preds  = self.softplus(logits)

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
        logits = clamp(logits,min=-1*max_ratio, max=max_ratio)
        
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
        #import pdb; pdb.set_trace()
        max_ratio=16
        logits = clamp(logits,min=-1*max_ratio, max=max_ratio)
        preds  = self.softplus(logits)
        
        y = torch.eye(preds.size(1))
        labels = y[labels].to(preds.device)

        use_logit_in  = labels     * (preds <  15).int()
        use_logit_out = (1-labels) * (preds > 1e-7).int()

        #import pdb; pdb.set_trace()


        inlier_loss  = clamp(use_logit_in * (1 - preds.pow(alpha))/(alpha), self.max_clip).sum(1)
        outlier_loss = clamp(use_logit_out * (preds.pow(1+alpha)-1)/(1+alpha), self.max_clip).mean(1)
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

    def get_modes(): return ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

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

class NormalLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(NormalLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        
    def get_modes(): return ['max', 'ce', 'norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.h(x)

class SoftplusLinear(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SoftplusLinear, self).__init__()

        self.h = nn.Linear(in_features, num_classes)
        self.init_weights()
        self.softplus = nn.Softplus()

    def get_modes(): return ['max', 'ce', 'norm']
    def init_weights(self):
        nn.init.kaiming_normal_(self.h.weight.data, nonlinearity = "relu")
        self.h.bias.data = torch.zeros(size = self.h.bias.size())

    def forward(self, x):
        return self.softplus(self.h(x))


class RatioLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super(RatioLayer, self).__init__()

        self.inlier  = nn.Linear(in_features, num_classes)
        self.outlier = nn.Linear(in_features, num_classes)
        self.in_featuressqrt = (in_features **.5)
        self.init_weights()

    def get_modes(): return ['logit_max','h_max','g_max', 'logit_ce','h_ce','g_ce', 'logit_norm', 'h_norm', 'g_norm']

    def init_weights(self):
        nn.init.kaiming_normal_(self.inlier.weight.data, nonlinearity = "relu")
        nn.init.kaiming_normal_(self.outlier.weight.data, nonlinearity = "relu")
        self.inlier.bias.data  = torch.zeros(size = self.inlier.bias.size())
        self.outlier.bias.data = torch.zeros(size = self.outlier.bias.size())

    def forward(self, x, max_ratio=15):
        max_ratio=15
        eps =  1e-6
        i = self.inlier(x) #/ self.in_featuressqrt
        o = self.outlier(x) #/ self.in_featuressqrt

        if self.training:
            i = clamp(i, min=-max_ratio, max=max_ratio)
            o = clamp(o, min=-max_ratio, max=max_ratio)
        #i = torch.tanh(i) * max_ratio
        #o = torch.tanh(o) * max_ratio
        #if self.training or True:
        #    i = i  / (torch.norm(self.inlier.weight ,p=2,dim=1)/2).unsqueeze(0)
        #    o = o / (torch.norm(self.outlier.weight,p=2,dim=1)/2).unsqueeze(0)

        #a = (torch.exp(o) + 1) * torch.exp(i - o)
        #b = 1+ torch.exp(i)
        #logits = a /b 

        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ eps)
        #logits = (1+torch.exp(-o)) / (1+torch.exp(-i))
        #if not self.training: logits = torch.clamp(logits ,min=-50000, max=50000)
        if self.training:
            return logits
        else:
            return logits, i, o


    def forwardold(self, x, max_ratio=50):
        eps =  .00000001
        i = self.inlier(x)
        o = self.outlier(x)
        if self.training or True:
            i = i  / (torch.norm(self.inlier.weight ,p=2,dim=1)/2).unsqueeze(0)
            o = o / (torch.norm(self.outlier.weight,p=2,dim=1)/2).unsqueeze(0)
        logits = torch.sigmoid(i) / (torch.sigmoid(o)+ eps)
        #if not self.training: logits = torch.clamp(logits ,min=-50000, max=50000)

        return logits
        if self.training:
            i = clamp(i, min=-max_ratio, max=max_ratio)
            o = clamp(o, min=-max_ratio, max=max_ratio)
            logits = torch.sigmoid(i) / (torch.sigmoid(o)+ eps)
            return logits#, o
        else:
            logits = torch.sigmoid(clamp(i ,min=-5000, max=5000)) / (torch.sigmoid(o+eps) +eps)
        
        return logits

CustomLayers = {
    'linear': NormalLinear,
    'godin': GodinLayer,
    'ratio':RatioLayer,
    'linearsp': SoftplusLinear,
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
    elif mode  in ['norm','h_norm','g_norm','logit_norm']:
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
            if 'logit_' in m:
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
