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
from resnet import ResNet34
from wideresnet import WideResNet
from deconfnet import DeconfNet, CosineDeconf, InnerDeconf, EuclideanDeconf, RatioDeconf
from datasets import OpenDatasets
#from collections import defaultdict

from generatingloaders import Normalizer, GaussianLoader, UniformLoader

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


h_dict = {
    'cosine':   CosineDeconf,
    'inner':    InnerDeconf,
    'euclid':   EuclideanDeconf,
    'ratio':   RatioDeconf
}


class losses:
    def __init__(self, do_softplus=True):
        self.softplus = nn.Softplus() if do_softplus else lambda x: x


    def kliep_loss(self, logits, labels, max_ratio=50):
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

generating_loaders_dict = {
    'Gaussian': GaussianLoader,
    'Uniform': UniformLoader
}

def get_args():
    parser = argparse.ArgumentParser(description='Pytorch Detecting Out-of-distribution examples in neural networks')
    
    # Device arguments
    parser.add_argument('--gpu', default = 0, type = int,
                        help = 'gpu index')

    # Model loading arguments
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--model-dir', default = './models', type = str,
                        help = 'model name for saving')

    # Architecture arguments
    parser.add_argument('--architecture', default = 'densenet', type = str,
                        help = 'underlying architecture (densenet | resnet | wideresnet)')
    parser.add_argument('--similarity', default = 'cosine', type = str,
                        help = 'similarity function for decomposed confidence numerator (cosine | inner | euclid | baseline)')
    parser.add_argument('--loss-type', default = 'ce', type = str,
                        help = 'ce|kliep')
    parser.add_argument('--opencat-exp', default = '', type = str,
                        help = 'ce|kliep')

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


def main():
    args = get_args()
    
    device           = args.gpu
    
    load_model       = args.load_model
    model_dir        = args.model_dir

    architecture     = args.architecture
    similarity       = args.similarity
    loss_type        = args.loss_type
    do_softplus      = args.do_softplus == 1
    
    data_dir         = args.data_dir
    data_name        = args.out_dataset
    batch_size       = args.batch_size
    #opencat_exp      = args.opencat_exp
    
    train            = args.train
    weight_decay     = args.weight_decay
    epochs           = args.epochs

    test             = args.test
    noise_magnitudes = args.magnitudes

   
    if args.opencat_exp != "":
        train_data, val_data, test_data, open_data, num_classes = get_opencat(data_dir, args.opencat_exp, batch_size)
    else:
        #get outlier data
        num_classes = 10
        train_data, val_data, test_data, open_data = get_datasets(data_dir, data_name, batch_size)

    # Create necessary directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if architecture == 'densenet':
        underlying_net = DenseNet3(depth = 100, num_classes = num_classes)
    elif architecture == 'resnet':
        underlying_net = ResNet34()
    elif architecture == 'wideresnet':
        underlying_net = WideResNet(depth = 28, num_classes = num_classes, widen_factor = 10)
    
    underlying_net.to(device)
    
    # Construct g, h, and the composed deconf net
    baseline = (similarity == 'baseline')

    if baseline:
        h = InnerDeconf(underlying_net.output_size, num_classes)
    else:
        h = h_dict[similarity](underlying_net.output_size, num_classes)

    h.to(device)

    deconf_net = DeconfNet(underlying_net, underlying_net.output_size, num_classes, h, baseline or similarity == "ratio")
    
    deconf_net.to(device)

    parameters = []
    h_parameters = []
    for name, parameter in deconf_net.named_parameters():
        if name == 'h.h.weight' or name == 'h.h.bias' or name == 'h.h1.weight' or name == 'h.h1.bias':
            h_parameters.append(parameter)
        else:
            parameters.append(parameter)

    optimizer = optim.SGD(parameters, lr = 0.1, momentum = 0.9, weight_decay = weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    h_wd = weight_decay if baseline else 0
    h_optimizer = optim.SGD(h_parameters, lr = 0.1, momentum = 0.9, weight_decay = h_wd) # No weight decay
    h_scheduler = optim.lr_scheduler.MultiStepLR(h_optimizer, milestones = [int(epochs * 0.5), int(epochs * 0.75)], gamma = 0.1)
    
    # Load the model (capable of resuming training or inference)
    # from the checkpoint file

    if load_model or (not train):
        checkpoint = torch.load(f'{model_dir}/checkpoint.pth')
        
        epoch_start = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        h_optimizer.load_state_dict(checkpoint['h_optimizer'])
        deconf_net.load_state_dict(checkpoint['deconf_net'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        h_scheduler.load_state_dict(checkpoint['h_scheduler'])
        epoch_loss = checkpoint['epoch_loss']
    else:
        epoch_start = 0
        epoch_loss = None


  
    criterion = losses(do_softplus)
    criterion = criterion.get_loss_dict()[loss_type]

    # Train the model
    if train:
        deconf_net.train()
        
        num_batches = len(train_data)
        epoch_bar = tqdm(total = num_batches * epochs, initial = num_batches * epoch_start)
        
        for epoch in range(epoch_start, epochs):
            total_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(train_data):
                if epoch_loss is None:
                    epoch_bar.set_description(f'{epoch + 1}/{epochs} | Batch {batch_idx + 1}/{num_batches}')
                else:
                    epoch_bar.set_description(f'{epoch + 1}/{epochs}| loss {epoch_loss:0.2f} | bat {batch_idx + 1}/{num_batches}')
                inputs = inputs.to(device)
                targets = targets.to(device)
                h_optimizer.zero_grad()
                optimizer.zero_grad()
                
                logits, _, _ = deconf_net(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                
                optimizer.step()
                h_optimizer.step()
                total_loss += loss.item()
                
                epoch_bar.update()
            
            epoch_loss = total_loss
            h_scheduler.step()
            scheduler.step()
            
            checkpoint = {
                'epoch': epoch + 1,
                'optimizer': optimizer.state_dict(),
                'h_optimizer': h_optimizer.state_dict(),
                'deconf_net': deconf_net.state_dict(),
                'scheduler': scheduler.state_dict(),
                'h_scheduler': h_scheduler.state_dict(),
                'epoch_loss': epoch_loss,
            }
            torch.save(checkpoint, f'{model_dir}/checkpoint.pth') # For continuing training or inference
            torch.save(deconf_net.state_dict(), f'{model_dir}/model.pth') # For exporting / sharing / inference only
        
        if epoch_loss is None:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Batch {num_batches}/{num_batches}')
        else:
            epoch_bar.set_description(f'Training | Epoch {epochs}/{epochs} | Epoch loss = {epoch_loss:0.2f} | Batch {num_batches}/{num_batches}')
        epoch_bar.close()

    #outfile = os.path.join("results", file_base + ".csv" )
    #with open(outfile, "a") as writer:
    #    writer.write(f"{args.seed}, {final_acc:.5f}, {final_auc:.5f}, {bays_auc:.5f}, {dr_auc:.5f}\n")

    if test:
        deconf_net.eval()
        best_val_score = None
        best_auc = None
        
        for noise_magnitude in noise_magnitudes:
            print(f'Noise magnitude {noise_magnitude:.5f}         ')
        
            for score_func in ['logit', 'h', 'g']:
                print(f'Score function: {score_func}')

        
                if val_data is not None:
                    validation_results =  np.average(testData(deconf_net, device, val_data, noise_magnitude, criterion, score_func, title = 'Validating'))
                else:
                    validation_results = 0
                print('ID Validation Score:',validation_results)
                
                id_test_results = testData(deconf_net, device, test_data, noise_magnitude, criterion, score_func, title = 'Testing ID', get_acc=True) 
                
                ood_test_results = testData(deconf_net, device, open_data, noise_magnitude, criterion, score_func, title = 'Testing OOD')
                auroc = calc_auroc(id_test_results, ood_test_results)*100
                tnrATtpr95 = calc_tnr(id_test_results, ood_test_results)
                print('AUROC:', auroc, 'TNR@TPR95:', tnrATtpr95)



                if best_auc is None:
                    best_auc = auroc
                else:
                    best_auc = max(best_auc, auroc)
                    print("best auc", best_auc)
                if best_val_score is None or validation_results > best_val_score:
                    best_val_score = validation_results
                    best_val_auc = auroc
                    best_tnr = tnrATtpr95
        
        print('supposedly best auc: ', best_val_auc, ' and tnr@tpr95 ', best_tnr)
        print('true best auc:'      , best_auc)

def calc_tnr(id_test_results, ood_test_results):
    scores = np.concatenate((id_test_results, ood_test_results))
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    fpr, tpr, thresholds = roc_curve(trues, scores)
    return 1 - fpr[np.argmax(tpr>=.95)]

def calc_auroc(id_test_results, ood_test_results):
    #calculate the AUROC
    scores = np.concatenate((id_test_results, ood_test_results))
    print(scores)
    trues = np.array(([1] * len(id_test_results)) + ([0] * len(ood_test_results)))
    result = roc_auc_score(trues, scores)

    return result

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
